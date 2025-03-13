import torch
import torch.nn as nn
from torch import tensor
import math
import torch.nn.functional as F
from target_search_network.encoder_residues import Encoder
from target_search_network.decoder_2D_skeletons import Decoder2D
from target_search_network.decoder_3D_skeletons import Decoder3D
from functools import partial
from timm.models.layers import DropPath
import numpy as np
from utils import to_long_tensor


device = torch.device('cuda:0')
print('The code uses GPU!!!')


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):    #归一化 Xi = (Xi-μ)/σ
    def __init__(self, emb_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(emb_size))
        self.beta = nn.Parameter(torch.zeros(emb_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

#用于将输入序列进行嵌入（Embedding）和添加位置编码的模块
class Embeddings_add_position(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings_add_position, self).__init__()
        self.emb_size = emb_size
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.LayerNorm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_dp):
        seq_length = input_dp.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_dp.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_dp).unsqueeze(2)

        words_embeddings = self.word_embeddings(input_dp)

        pe = torch.zeros(words_embeddings.shape).cuda()
        div = torch.exp(torch.arange(0., self.emb_size, 2) * -(math.log(10000.0) / self.emb_size)).double().cuda()
        pe[..., 0::2] = torch.sin(position_ids * div)
        pe[..., 1::2] = torch.cos(position_ids * div)

        embeddings = words_embeddings + pe
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

#用于将输入序列进行嵌入（Embedding）但不添加位置编码的模块
class Embeddings_no_position(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings_no_position, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.LayerNorm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self,input_dp):
        words_embeddings = self.word_embeddings(input_dp)
        embeddings = self.LayerNorm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


#为输入的药物三维特征添加位置信息
class Drug_3D_feature(nn.Module):
    def __init__(self, emb_size, p, device):
        self.emb_size = emb_size
        self.p = p #是一个比例参数，用于对位置编码进行缩放
        self.device = device
        super(Drug_3D_feature, self).__init__()

    def forward(self, input_dp, pos):
        item = []
        pos = pos * self.p

        # sum PE with token embeddings
        div = torch.exp(torch.arange(0., self.emb_size, 2) * -(math.log(10000.0) / self.emb_size)).double().to(self.device)
        for i in range(3):
            pe = torch.zeros(input_dp.shape).to(self.device)
            pe[..., 0::2] = torch.sin(pos[..., i].unsqueeze(2) * div)
            pe[..., 1::2] = torch.cos(pos[..., i].unsqueeze(2) * div)
            item.append(pe)
        input_dp = input_dp.unsqueeze(1)
        drug_3d_feature = input_dp
        for i in range(0, 3):
            channel_i = input_dp + item[i].unsqueeze(1)
            drug_3d_feature = torch.cat([drug_3d_feature, channel_i], dim=1)
        return drug_3d_feature


#计算药物中每个原子的特征
class GAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size = Wh.size()[0]
        N = Wh.size()[1]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).reshape(batch_size, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        item = all_combinations_matrix.view(batch_size, N, N, 2 * self.out_features)
        return item

    def forward(self, atoms_vector, adjacency):
        Wh = torch.matmul(atoms_vector, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)         # 连接操作
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), 0.1)

        zero_vec = -9e15 * torch.ones_like(e)                           # 极小值
        attention = torch.where(adjacency > 0, e, zero_vec)             # 不相连的用极小值代替
        attention = F.softmax(attention, dim=2)                         # 权重系数α_ij，不想连的趋近于零
        h_prime = torch.bmm(attention, Wh)
        return F.elu(h_prime)

'''
 Attention 类实现了多头自注意力模块的计算过程，可以用于构建具有自注意力机制的神经网络层
'''
class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,  # 生成qkv 是否使用偏置
                 drop_ratio=0.,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 #它是一个缩放因子，用于在计算自注意力分数时对头维度进行缩放，以控制分数的范围。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)  #用于将自注意力的输出映射回原始特征维度 dim。
        self.proj_drop = nn.Dropout(drop_ratio)

    #自注意力模块的前向传播函数
    def forward(self, x):
        '''
        B：批量大小（Batch Size）
        N：序列长度（Number of elements in the sequence）
        C：每个元素的特征维度（Channel）
        '''
        B, N, C = x.shape

        # reshape是用于改变张量形状的操作
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对注意力分数进行softmax操作，将其转换为概率分布。
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        #输出的x的维度是 (B, N, C)
        return x


#带有残差连接的注意力Block
class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim) #对输入进行归一化操作
        #自注意力机制
        self.attn = Attention(dim, drop_ratio=drop_ratio)

        # 用于在训练过程中以一定概率随机丢弃一部分注意力计算的结果，以防止过拟合。
        # nn.Identity()是PyTorch中的一个恒等映射层。该层的前向传播操作是输入张量的恒等映射，即输出与输入相同
        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) #对输入进行归一化操作

        # 实例化MLp模型
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x):
        # 两次残差连接
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Drug_feature(nn.Module):
    def __init__(self,
                 depth_e2=4,
                 embed_dim=256, # 128
                 drop_ratio=0.,
                 device='cuda:0'
                 ):
        super(Drug_feature, self).__init__()
        self.device = device
        #用于创建具有一组通用参数的可重复使用的函数或类。
        # 在这个特定的情况下，nn.LayerNorm 是一个用于归一化神经网络层的类，eps 是一个控制数值稳定性的参数。
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # self.rate1 = torch.nn.Parameter(torch.rand(1))
        # self.rate2 = torch.nn.Parameter(torch.rand(1))


        # 猜测药物的文本特征
        self.smile2feature = nn.Linear(512, 256)
        self.embeddings_e2 = nn.Embedding(51, embed_dim)
        self.norm_e2 = norm_layer(embed_dim)
        self.pos_drop_e2 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e2 = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr_e2 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e2)]
        self.encoder_e2 = nn.Sequential(*[
            #MSA
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e2[i],
                  )
            for i in range(depth_e2)
        ])

        self.feature_backbone = nn.Sequential(
            # 待添加代码
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0., inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0., inplace=True),

        )

    #处理药物的文本特征
    def forward_features_e2(self, x):
        x = self.embeddings_e2(x)
        x = x.permute(0, 2, 1)
        x = self.smile2feature(x)
        x = x.permute(0, 2, 1)
        # 位置嵌入+位置信息（我猜的！！！！！！！！！）
        x = self.pos_drop_e2(x + self.pos_embed_e2)
        #嵌入信息 MSA
        x = self.encoder_e2(x)
        x = self.norm_e2(x)
        return x


    def forward(self, smile):
        # smile = torch.LongTensor(smile)
        # smile = to_long_tensor(smile)
        smile = smile.to(self.device)
        smile_feature = self.forward_features_e2(smile)
        # smile_feature = self.feature_backbone(smile_feature)
        return smile_feature

class Drug_image(nn.Module):
    def __init__(self,
                 depth_e1=4,
                 embed_dim=256,
                 drop_ratio=0.,
                 backbone="CNN",
                 device='cuda:0',
                 batch_size=128,
                 ):
        super(Drug_image, self).__init__()

        self.device = device
        self.batch_size = batch_size

        #用于创建具有一组通用参数的可重复使用的函数或类。
        # 在这个特定的情况下，nn.LayerNorm 是一个用于归一化神经网络层的类，eps 是一个控制数值稳定性的参数。
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        #，"CNN" 表示使用卷积神经网络作为骨干网络来处理图像数据
        if backbone == "CNN":
            # 实例化一个处理药物图像的模块
            self.img_backbone = nn.Sequential(  # 3*256*256
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # 64*128*128
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.02, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*64

                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),  # 256*32*32
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.02, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 256*16*16
            )

        #  encoder 1  处理图像数据
        self.norm_e1 = norm_layer(embed_dim)
        self.pos_drop_e1 = nn.Dropout(p=drop_ratio)
        #用于存储位置编码（Positional Embedding）。
        # 位置编码是一种用于在自注意力机制中考虑输入序列元素之间的相对位置信息的技巧。
        # 这里初始化了一个全零的位置编码，它将与输入数据相加以引入位置信息。
        self.pos_embed_e1 = nn.Parameter(torch.zeros(1, 256, embed_dim))
        #创建一个包含depth_e1个块（Block）的编码器（encoder_e1），每个块具有不同的丢弃率（dropout rate）。
        #包含了 depth_e1 个丢弃率的值
        dpr_e1 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e1)]
        self.encoder_e1 = nn.Sequential(*[
            # MSA
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e1[i],
                  )
            for i in range(depth_e1)
        ])

        self.img_backbone1 = nn.Sequential(
            # 待添加代码
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0., inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0., inplace=True),
        )

    #处理药物的图像
    def forward_features_e1(self, x):
        B, _, _, _ = x.shape
        #药物图像CNN卷积
        x = self.img_backbone(x)
        # x = x.to(self.device)
        x = x.reshape(B, 256, -1)
        #药物图像的位置嵌入+位置信息（我猜的！！！！！！！！！）
        x = self.pos_drop_e1(x + self.pos_embed_e1)
        #MSA
        x = self.encoder_e1(x)
        x = self.norm_e1(x)
        return x

    def forward(self, image):
        image = image.to(self.device)
        # print('1111111111', image.shape) #torch.Size([16, 3, 256, 256])
        image_feature = self.forward_features_e1(image)
        # print('2222222222', image_feature.shape) #torch.Size([16, 256, 256])
        image_feature = self.img_backbone1(image_feature)
        # print('33333333333', image_feature.shape)
        # print('33333333333', image_feature.dtype)
        # 对输入图像进行二维离散傅里叶变换
        image_feature = torch.fft.fft2(image_feature, dim=(-2, -1))
        # image_feature = image_feature.real.float()
        image_feature = torch.fft.fftshift(image_feature, dim=(-2, -1))
        image_feature = image_feature.real.float()
        # print("44444444444:", image_feature.shape)
        # print("44444444444:", image_feature.dtype)
        return image_feature


# ------------------------------------再另一篇论文中  加入这个模块---------------
class Drug_fusion(nn.Module):
    def __init__(self,
                 device='cuda:0',
                 input_size=256,
                 input_size_2=128,
                 output_size=64,
                 batch_size=16,
                 flatten_dim=64,
                 dropout_rate=0.2):
        super(Drug_fusion, self).__init__()
        self.batch_size = batch_size
        self.flatten_dim = flatten_dim
        self.pool = nn.AvgPool1d(4)
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_size, output_size)  # 用于学习权重的线性层
        self.fc2 = nn.Linear(input_size_2, output_size)  # 用于学习权重的线性层
        scale_ = input_size // output_size
        self.scale = scale_ ** -0.5  # 它是一个缩放因子，用于在计算自注意力分数时对头维度进行缩放，以控制分数的范围。


    def forward(self, drug_1, drug_2):

        # --------------------------特征融合模块-----------------------------------
        drug_1 = drug_1.to(device)
        drug_2 = drug_2.to(device)

        # 归一化输入张量
        drug_2 = F.normalize(drug_2, p=2, dim=-1)
        drug_1 = F.normalize(drug_1, p=2, dim=-1)

        affinity_matrix = torch.matmul(drug_1, drug_2.transpose(-1, -2))

        # affinity_scores = affinity_matrix / math.sqrt(12)
        affinity_scores = affinity_matrix * self.scale
        # print("affinity_scores:", affinity_scores.shape) #torch.Size([16, 16])

        # Normalize the attention scores to probabilities.
        drug1_scores = nn.Softmax(dim=-1)(affinity_scores)
        # print("drug_scores:", drug_scores.shape) #torch.Size([16, 16])

        drug2_scores = nn.Softmax(dim=-2)(affinity_scores)
        # print("protein_scores:", protein_scores.shape) #torch.Size([16, 16])

        drug1_layer = torch.matmul(drug2_scores.transpose(-1, -2), drug_1)
        # print("drug_layer:", drug_layer.shape) #torch.Size([16, 64])

        drug2_layer = torch.matmul(drug1_scores, drug_2)
        # print("protein_layer:", protein_layer.shape) #torch.Size([16, 64])


        drug1_emb = torch.cat((drug_1, drug2_layer), dim=-1)
        # print("drug_emb:", drug_emb.shape) #torch.Size([16, 128])

        drug2_emb = torch.cat((drug_2, drug1_layer), dim=-1)
        # print("protein_emb:", protein_emb.shape) #torch.Size([16, 128])

        # # 添加的代码
        # drug1_emb = F.relu(self.fc2(drug1_emb))
        # drug1_emb = F.dropout(drug1_emb, p=self.dropout_rate)
        # # print("drug1_emb:", drug1_emb.shape) # torch.Size([16, 64])
        # drug2_emb = F.relu(self.fc2(drug2_emb))
        # drug2_emb = F.dropout(drug2_emb, p=self.dropout_rate)
        # # print("drug2_emb:", drug2_emb.shape) # torch.Size([16, 64])
        #
        # return drug1_emb, drug2_emb


        fusion_emb = torch.cat((drug1_emb, drug2_emb), dim=1)
        # print("fusion_emb:", fusion_emb.shape) #torch.Size([16, 256])

        fusion_emb = F.dropout(fusion_emb, p=self.dropout_rate)
        # print("111111:", fusion_emb.shape)

        # 应用学到的权重
        fusion_weights = F.relu(self.fc1(fusion_emb))
        # print("2222222:", fusion_weights.shape) # torch.Size([16, 64])
        fusion_weights = F.softmax(fusion_weights, dim=-1)
        # print("3333333:", fusion_weights.shape)

        fusion_emb = self.pool(fusion_emb)

        # 将学到的权重应用于 fusion_emb
        fusion_emb_weighted = fusion_emb * fusion_weights
        # print("44444444:", fusion_emb_weighted.shape)

        return fusion_emb_weighted


# ------------------------在另一个论文中引用交叉注意力机制（药物文本和图像特征）----------------------------------

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_out(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,  # 生成qkv 是否使用偏置
                 drop_ratio=0.,
                 ):
        super(Attention_out, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 #它是一个缩放因子，用于在计算自注意力分数时对头维度进行缩放，以控制分数的范围。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)  #用于将自注意力的输出映射回原始特征维度 dim。
        self.proj_drop = nn.Dropout(drop_ratio)

    #自注意力模块的前向传播函数
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#MCA Block
class Cross_Attention(nn.Module):
    def __init__(self,
                 dim, #输入特征的维度
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.2,
                 ):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads #每个头的维度
        self.scale = head_dim ** -0.5 #用于缩放注意力分数。开方操作
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #将 输入特征 映射到查询（query）、键（key）、值（value）的空间
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim) #投影层：将注意力计算的输出进行进一步的线性变换
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x, y): #两个输入张量 x 和 y，分别表示查询和键-值对。
        B, N, C = x.shape
        #reshape 操作将结果变形为 [B, N, 3, num_heads, C // num_heads] 的张量，
        # 其中 B 表示批量大小，N 表示序列长度，C 表示输入特征维度。然后使用 permute 操作将维度重新排列，
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv2[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.2,
                 ):
        super(Decoder, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_out(dim, drop_ratio=drop_ratio)
        self.cross_attn = Cross_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x, y):
        y = y + self.drop_path(self.attn(self.norm1(y)))
        out = y + self.drop_path(self.cross_attn(x, self.norm1(y)))
        out = out + self.drop_path(self.mlp(self.norm2(y)))
        return out
# ---------------------------------------------------------------------------------------------

class MdDTI(nn.Module):
    def __init__(self, l_drugs_dict, l_proteins_dict, l_substructures_dict, args):
        super(MdDTI, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.task = args.task.upper()
        self.p = args.p
        self.emb1_size = args.emb1_size
        self.emb2_size = args.emb2_size
        self.drug_dim = args.drug_emb_dim
        self.target_dim = args.target_emb_dim
        self.feature_model = Drug_feature()
        self.drug_self_attention_block = Block(dim=128, mlp_ratio=4, drop_ratio=0.)
        self.image_model = Drug_image()
        self.target_self_attention_block = Block(dim=256, mlp_ratio=4, drop_ratio=0.)
        self.drug_fusion = Drug_fusion()
        self.device = args.device
        self.flat = nn.Flatten()
        self.linear_layer_feature = nn.Linear(256*256, 64)  # 调整药物特征维度的线性层
        self.linear_layer_image = nn.Linear(256 * 256, 64)  # 调整药物图像维度的线性层
        # self.rate1 = torch.nn.Parameter(torch.rand(1))
        self.rate1 = torch.nn.Parameter(0.3 * torch.rand(1) + 0.55)
        self.rate2 = torch.nn.Parameter(1 - self.rate1)
        # self.rate2 = torch.nn.Parameter(torch.rand(1))
        # print("rate1:", self.rate1)
        # print("rate2:", self.rate2)


        # 检查并调整参数顺序
        # self.check_parameter_order()

        # target
        #对靶标分割后的残基进行嵌入操作
        self.embedding_residues = Embeddings_add_position(l_proteins_dict + 1, self.target_dim)
        self.encoder_residues = Encoder(self.target_dim, 2, 1024)

        # drug 2D substructure
        #对药物的子结构进行嵌入操作
        self.embedding_substructures = Embeddings_add_position(l_substructures_dict + 1, self.drug_dim)
        self.Decoder2D = Decoder2D(self.drug_dim, 2, 512)

        # drug 3D atom
        #对药物的空间结构进行嵌入操作
        self.embedding_skeletons = Embeddings_no_position(l_drugs_dict + 1, self.drug_dim)
        self.gat1 = GAT(self.drug_dim, self.drug_dim)
        self.gat2 = GAT(self.drug_dim, self.drug_dim)
        self.drug_3d_feature = Drug_3D_feature(self.drug_dim, self.p, self.device)
        self.channel_max_pool = nn.MaxPool1d(4)
        self.channel_avg_pool = nn.AvgPool1d(4)
        self.spatial_conv2d = nn.Conv2d(2, 1, (5, 5), padding=2)
        self.bn = nn.BatchNorm2d(4, eps=1e-5, momentum=0.01, affine=True)
        self.α = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.constant_(self.α.data, 0.5)
        self.Decoder3D = Decoder3D(self.drug_dim, 2, 1024)

        self.ln = nn.LayerNorm(self.drug_dim)

        # predict 预测模块
        if self.task == 'DTI':
            self.predict = nn.Sequential(
                nn.Linear(self.drug_dim + self.target_dim, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 32),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

        elif self.task == 'CPA':
            self.predict = nn.Sequential(
                nn.Linear(self.drug_dim + self.target_dim, 1024),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(1024, 64),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        # 添加原代码的靶标的一维卷积层
        self.Target_1D_Convolution = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        # 添加原代码的药物的一维卷积层
        self.Drug_1D_Convolution = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        # ------------------------在另一个论文中引用交叉注意力机制-------------------
        # -------------------添加的代码-----------------------------------
        self.decoder1_1 = Decoder(self.emb1_size, mlp_ratio=4, )
        self.decoder2_1 = Decoder(self.emb1_size, mlp_ratio=4, )

        self.decoder1_2 = Decoder(self.emb2_size, mlp_ratio=4, )
        self.decoder2_2 = Decoder(self.emb2_size, mlp_ratio=4, )
        self.depth_decoder = 4
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm_decoder1 = norm_layer(self.emb1_size)
        self.norm_decoder2 = norm_layer(self.emb2_size)

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1),  # 保持相同的空间尺寸
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 减小空间尺寸

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 保持相同的空间尺寸
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 减小空间尺寸
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc1 = nn.Linear(992, 64)
        self.fc2 = nn.Linear(992, 2)
        self.fc3 = nn.Linear(16352, 64)

        # MCA

    def forward_features_decoder1(self, x, y):

        B, _, _ = x.shape

        # cross1
        # decoder==>MCA
        out1 = self.decoder1_1(x, y)
        for i in range(self.depth_decoder - 1):
            out1 = self.decoder1_1(x, out1)
        out1 = self.norm_decoder1(out1)
        out1 = out1.reshape(B, 1, 256, -1)

        # cross2
        # decoder2==>MCA
        out2 = self.decoder2_1(y, x)
        for i in range(self.depth_decoder - 1):
            out2 = self.decoder2_1(y, out2)
        out2 = self.norm_decoder1(out2)
        # print("out2_1:",out2.shape) #
        out2 = out2.reshape(B, 1, 256, -1)
        # print("out2_2:", out2.shape) #
        out = torch.cat((out1, out2), 1)
        # print("out:",out.shape) #
        return out

    def forward_features_decoder2(self, x, y):

        B, _, _ = x.shape

        # cross1
        # decoder==>MCA
        out1 = self.decoder1_2(x, y)
        for i in range(self.depth_decoder - 1):
            out1 = self.decoder1_2(x, out1)
        out1 = self.norm_decoder2(out1)
        out1 = out1.reshape(B, 1, 64, -1)

        # cross2
        # decoder2==>MCA
        out2 = self.decoder2_2(y, x)
        for i in range(self.depth_decoder - 1):
            out2 = self.decoder2_2(y, out2)
        out2 = self.norm_decoder2(out2)
        # print("out2_1:",out2.shape) # torch.Size([16, 64, 64])
        out2 = out2.reshape(B, 1, 64, -1)
        # print("out2_2:", out2.shape) # torch.Size([16, 1, 64, 64])
        out = torch.cat((out1, out2), 1)
        # print("out:",out.shape) # torch.Size([16, 2, 64, 64])
        return out

    # -----------------------------------------------------------------------------------

    def p_value(self, x):
        if x < 0.2:
            return 0.2
        elif x > 0.8:
            return 0.8
        else:
            return x

    #提取出重原子特征
    def extract_heavy_atoms_feature(self, atoms_feature, marks, padding_tensor):
        lens = [int(sum(mark)) for mark in marks]
        max_len = max(lens)

        heavy_atoms_f = torch.zeros(atoms_feature.shape[0], max_len, atoms_feature.shape[2])
        heavy_atoms_masks = torch.ones(atoms_feature.shape[0], max_len)  # Mark the heavy atoms and filled atoms.
        for i in range(len(lens)):
            heavy_atoms_f[i, :, :] = padding_tensor
            heavy_atoms_masks[i, lens[i]:] = 0

        for i, mark in enumerate(marks):
            indices = torch.nonzero(mark).squeeze()
            if indices.dim() == 0:   # The case of only containing one heavy atom.
                indices = indices.unsqueeze(0)
            heavy_atoms_f[i, :indices.shape[0]] = atoms_feature[i, indices]

        heavy_atoms_f = heavy_atoms_f.to(self.device)
        heavy_atoms_masks = heavy_atoms_masks.to(self.device)
        return heavy_atoms_f, heavy_atoms_masks

    #提取出药物的空间结构特征 式8
    def spatial_attention(self, skeletons_3d_feature):
        spatial_att_max, spatial_att_avg = None, None
        for i in range(self.batch_size):
            skeletons_feature = skeletons_3d_feature[i].permute(1, 2, 0)
            channel_max_pool = self.channel_max_pool(skeletons_feature).unsqueeze(0)
            channel_avg_pool = self.channel_avg_pool(skeletons_feature).unsqueeze(0)
            if i == 0:
                spatial_att_max = channel_max_pool
                spatial_att_avg = channel_avg_pool
            else:
                spatial_att_max = torch.cat([spatial_att_max, channel_max_pool], dim=0)
                spatial_att_avg = torch.cat([spatial_att_avg, channel_avg_pool], dim=0)
        spatial_att = self.spatial_conv2d(torch.cat([spatial_att_max, spatial_att_avg], dim=3).permute(0, 3, 1, 2))
        skeletons_3d_feature = skeletons_3d_feature * torch.sigmoid(spatial_att)
        return skeletons_3d_feature
    '''
    #一致性损失函数
    def consistency_loss(self, data1, data2):
        loss = 0
        for i in range(self.batch_size):
            d1 = data1[i].reshape(1, -1)
            d2 = data2[i].reshape(1, -1)
            l_upper = torch.mm(d1, d2.T)
            l_down = torch.mm(torch.sqrt_(torch.sum(d1.mul(d1), dim=1).view(torch.sum(d1.mul(d1), dim=1).shape[0], 1)),
                              torch.sqrt_(torch.sum(d2.mul(d2), dim=1).view(torch.sum(d2.mul(d2), dim=1).shape[0], 1)))
            score = torch.div(l_upper, l_down)
            loss = loss + 1 - score[0][0]
        loss = loss / self.batch_size
        return loss
    '''

    def consistency_loss(self, data1, data2, data3):
        loss = 0

        for i in range(self.batch_size):
            d1 = data1[i].reshape(1, -1)
            d2 = data2[i].reshape(1, -1)
            d3 = data3[i].reshape(1, -1)

            # 计算data1和data2之间的一致性得分
            l_upper_1 = torch.mm(d1, d2.T)
            l_down_1 = torch.sqrt(torch.sum(d1.mul(d1), dim=1)).view(-1, 1) * torch.sqrt(
                torch.sum(d2.mul(d2), dim=1)).view(1, -1)
            score_1 = torch.div(l_upper_1, l_down_1)

            # 计算data1和data3之间的一致性得分
            l_upper_2 = torch.mm(d1, d3.T)
            l_down_2 = torch.sqrt(torch.sum(d1.mul(d1), dim=1)).view(-1, 1) * torch.sqrt(
                torch.sum(d3.mul(d3), dim=1)).view(1, -1)
            score_2 = torch.div(l_upper_2, l_down_2)

            # 计算data2和data3之间的一致性得分
            l_upper_3 = torch.mm(d2, d3.T)
            l_down_3 = torch.sqrt(torch.sum(d2.mul(d2), dim=1)).view(-1, 1) * torch.sqrt(
                torch.sum(d3.mul(d3), dim=1)).view(1, -1)
            score_3 = torch.div(l_upper_3, l_down_3)

            # 合并一致性得分
            total_score = score_1 + score_2 + score_3

            # 基于合并的得分更新损失
            loss = loss + torch.abs(1 - total_score[0][0])

        # 对整个批次的损失进行平均
        loss = loss / self.batch_size
        return loss

    # 计算药物文本特征和药物图像之间的损失

    def drug_text_image_loss(self, data1, data2):
        loss = 0
        for i in range(self.batch_size):
            d1 = data1[i].view(1, -1)
            d2 = data2[i].view(1, -1)

            # Calculate L2 norm consistency loss
            l2_loss = torch.nn.functional.mse_loss(d1, d2)

            # Accumulate loss for each sample
            loss += l2_loss.item()

        # Average the loss
        loss = loss / self.batch_size

        return loss

    '''
    def drug_text_image_loss(self, data1, data2):

        loss = 0
        for i in range(self.batch_size):
            d1 = data1[i].view(1, -1)
            d2 = data2[i].view(1, -1)

            # 计算余弦相似度
            l2_loss = 1 - F.cosine_similarity(d1, d2)

            # Accumulate loss for each sample
            loss += l2_loss.item()

        # Average the loss
        loss = loss / self.batch_size

        return loss
    '''

    # def check_parameter_order(self):
    #     # 确保 rate1 > rate2
    #     if self.rate1 <= self.rate2:
    #         self.rate1.data = self.rate2.data + 0.1  # 调整rate1的值，确保大于rate2


    def forward(self, drug, target):

        residues, residues_masks = target['residues'], target['residues_masks']
        # print("residues:", residues.shape)  # torch.Size([24, 1000])
        # print("residues_masks:", residues_masks.shape)  # torch.Size([24, 1000])

        target_second_structures, target_second_structures_masks = target['tss'], target['tss_masks']
        # print("target_second_structures:", target_second_structures.shape)  #
        # print("target_second_structures_masks:", target_second_structures_masks.shape)  #

        substructures, substructures_masks = drug['substructures'], drug['substructures_masks']
        # print("substructures:", substructures.shape) #torch.Size([24, 15])
        # print("substructures_masks:", substructures_masks.shape) # torch.Size([24, 15])

        skeletons, adjs, marks = drug['skeletons'], drug['adjs'], drug['marks']
        positions, padding_tensor_idx = drug['positions'], drug['padding_tensor_idx']

        # 添加药物的文本特征
        drug_text, drug_text_mask = drug['drug_features'], drug['drug_features_mask']
        # print("drug_feature:", drug_text.shape) # torch.Size([24, 512])
        # print("drug_feature_mask:", drug_text_mask.shape) # torch.Size([24, 512])

        #添加药物图像信息
        drug_image = drug['drug_images']  # list
        drug_image = torch.stack(drug_image, dim=0)

        # drug images
        drug_image = self.image_model(drug_image)
        # print("333333333333:", image_feature)
        # print("444444444444", image_feature.shape) #torch.Size([16, 256, 256])
        # 添加一个自注意力模块 效果不好
        # image_feature = self.target_self_attention_block(image_feature) #torch.Size([16, 256, 256])
        # print("添加一个自注意力模块的维度：", image_feature.shape)
        drug_image = self.flat(drug_image)
        # print("555555555555:", image_feature.shape) #torch.Size([16, 65536])
        drug_image = self.linear_layer_image(drug_image)
        # print("66666666666:", drug_image.shape) #torch.Size([B, 64])

        # image_feature = (image_feature - torch.min(image_feature)) / (
        #                torch.max(image_feature) - torch.min(image_feature))

        # target
        #蛋白质的残基序列进行嵌入
        residues_emb = self.embedding_residues(residues)
        #Et放在靶标搜索网络(Decoder3D)里,得到ft:residues_feature
        residues_feature, K, V = self.encoder_residues(residues_emb, residues_masks)
        # print("residues_feature:",residues_feature.shape) #torch.Size([16, 1000, 64])

        # 蛋白质二级结构序列进行嵌入
        target_second_structures_emb = self.embedding_residues(target_second_structures)
        # Et放在靶标搜索网络(Decoder3D)里,得到ft:target_second_structures_feature
        target_second_structures_feature, K2, V2 = self.encoder_residues(target_second_structures_emb, target_second_structures_masks)
        # print("target_second_structures_feature:",target_second_structures_feature.shape) #


        # ------------------------------药物文本的嵌入---------------------------------------
        drug_text_emb = self.embedding_substructures(drug_text)
        # print("drug_text_emb:", drug_text_emb.shape)  #
        drug_text_feature, drug_text_feature_att = self.Decoder2D(drug_text_emb, drug_text_mask, K, V, residues_masks)
        # print("drug_text_feature:", drug_text_feature.shape) #张量时刻变化
        drug_text = self.ln(torch.amax(drug_text_feature, dim=1))
        # print("drug_text:",drug_text.shape) #torch.Size([B, 64])
        # ----------------------------------------------------------------------------------------


        # 将药物文本特征与药物图像特征进行融合(暂时先不用)
        drug_text_image = self.drug_fusion(drug_text, drug_image)
        # drug_text_image = 0.5 * (self.rate1 * drug_text + self.rate2 * drug_image) #效果非常非常差
        # print("drug_text_image:", drug_text_image.shape)  #torch.Size([B, 64])

        # drug 2d substructure
        #标题3.1 药物的二维子结构进行嵌入
        substructures_emb = self.embedding_substructures(substructures)
        # print("substructures_emb:",substructures_emb.shape) # torch.Size([24, 14, 64]) 时刻变化的
        #使用 Decoder2D 解码药物的二维亚结构，得到特征（substructures_feature）以及注意力相关的中间结果（substructures_att）。
        #Db放在靶标搜索网络(Decoder3D)里,得到fb:substructures_feature
        #药物的二维亚结构注意力分布：substructures_att
        substructures_feature, substructures_att = self.Decoder2D(substructures_emb, substructures_masks, K, V, residues_masks)
        # print("substructures_feature:", substructures_feature.shape) #张量时刻变化

        # drug 3D atom
        #标题3.2 药物的三维结构特征提取
        atoms_emb = self.embedding_skeletons(skeletons)
        padding_tensor = atoms_emb[padding_tensor_idx][-1]
        #3.2.1 进行 Graph Attention Network（GAT） 操作，包括两个 GAT 层（gat1 和 gat2），以得到原子的特征。
        atoms_feature = self.gat1(atoms_emb, adjs)
        atoms_feature = self.gat2(atoms_feature, adjs)

        #3.2.2 提取出药物中的重原子特征
        skeletons_feature, skeletons_masks = self.extract_heavy_atoms_feature(atoms_feature, marks, padding_tensor)
        #3.2.3 构造空间结构的特征矩阵
        skeletons_feature_3d = self.drug_3d_feature(skeletons_feature, positions)
        #3.2.4 提取药物的空间结构特征 式8
        spatial_feature = self.spatial_attention(skeletons_feature_3d)
        #式9
        skeletons_feature_3d = torch.sum(self.bn(spatial_feature), dim=1)
        #式10  skeletons_feature_3d：Dp;   skeletons_feature:Dg
        #此时这个位置的 skeletons_feature_3d 代表的是靶标的三维空间特征
        skeletons_feature_3d = skeletons_feature * self.p_value(self.α) + skeletons_feature_3d * (1 - self.p_value(self.α))
        #Ds放在靶标搜索网络(Decoder3D)里,得到fs:skeletons_feature_3d
        #药物的空间结构的注意力分布：skeletons_att
        skeletons_feature_3d, skeletons_att = self.Decoder3D(skeletons_feature_3d, skeletons_masks, K, V, residues_masks)
        # print("skeletons_feature_3d:", skeletons_feature_3d.shape) #张量时刻变化


        # 3.4 predict
        target_feature = self.ln(torch.amax(residues_feature, dim=1)) #torch.Size([B, 64])

        #蛋白质二级结构信息
        target_second_structures_feature = self.ln(torch.amax(target_second_structures_feature, dim=1))  #
        # print("target_second_structures_feature:", target_second_structures_feature.shape) #torch.Size([B, 64])

        #将药物分解后的子结构特征向量与药物的三维结构特征向量相加
        drug_feature = 0.5*(self.ln(torch.amax(substructures_feature, dim=1)) +
                            self.ln(torch.amax(skeletons_feature_3d, dim=1)))

       
        drug_feature = 0.5 * (self.rate1 * drug_feature + self.rate2 * drug_text_image)
        # print("drug_feature2:", drug_feature.shape) #torch.Size([B, 64])

        # 将靶标的一级序列和二级序列进行权重比的相加 效果可以
        target_feature = 0.5 * (self.rate1 * target_feature + self.rate2 * target_second_structures_feature)
        # print("target_feature2:", target_feature.shape) #torch.Size([B, 64])

        drug_target = torch.cat([target_feature, drug_feature], dim=1)
        # print("drug_target:", drug_target.shape)  #torch.Size([B, 128])
        # 药物-靶标预测结果：out
        out = self.predict(drug_target)
        # print("out:", out.shape) # torch.Size([B, 2])

        # 3.5 consistency loss
        #substructures_att：药物的二维亚结构注意力分布
        #skeletons_att：药物的空间结构的注意力分布
        # loss2 = self.consistency_loss(substructures_att, skeletons_att)
        loss2 = self.consistency_loss(substructures_att, skeletons_att, drug_text_feature_att)
        # loss2 = loss2 + loss1

        # x = skeletons_att[-1][:56].clone()
        # y = substructures_att[-1][:56].clone()
        # print("atts")
        # print(list(np.array(x.cpu())))
        # print(list(np.array(y.cpu())))

        # 输出结果：out:预测值
        #         loss2：药物-靶标预测损失
        return out, loss2, self.rate1, self.rate2
