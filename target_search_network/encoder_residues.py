import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,emb_size, num_attention_heads):
        super(SelfAttention,self).__init__()
        if emb_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads # 存储注意力头的数量
        self.attention_head_size = int(emb_size / num_attention_heads) # 计算每个注意力头的特征维度。
        self.all_head_size = self.num_attention_heads * self.attention_head_size #计算所有注意力头的总特征维度。

        self.query = nn.Linear(emb_size, self.all_head_size)
        self.key = nn.Linear(emb_size, self.all_head_size)
        self.value = nn.Linear(emb_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    #用于将线性变换的输出重新排列成适合注意力计算的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,dp_emb, dp_mask):
        #根据dp_emb来求的 Q K V
        mixed_query_layer = self.query(dp_emb)
        mixed_key_layer = self.key(dp_emb)
        mixed_value_layer = self.value(dp_emb)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        #
        attention_scores = attention_scores - dp_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        #
        return context_layer, key_layer, value_layer


class Attention(nn.Module):
    def __init__(self,emb_size, num_attention_heads):
        super(Attention,self).__init__()
        #emb_size 表示输入特征的维度，num_attention_heads 表示注意力头的数量。
        #实例化模型
        self.selfAttention_common = SelfAttention(emb_size, num_attention_heads)
        self.attentonLayer = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(emb_size)

    def forward(self,dp_emb, dp_mask):
        dp_mask = dp_mask.unsqueeze(1).unsqueeze(2) #使其变为一个四维张量
        dp_mask = (1.0 - dp_mask) * 100000.0
        selfAttention_output, K, V = self.selfAttention_common(dp_emb, dp_mask)
        attentionLayer_output = self.attentonLayer(selfAttention_output)
        attentionLayer_dropout = self.dropout(attentionLayer_output)
        attention_output = self.LayerNorm(attentionLayer_dropout + dp_emb) #我理解是残差连接
        return attention_output, K, V

#Transformer的编码层
class Encoder(nn.Module):
    def __init__(self, emb_size, num_attention_heads, hidden_size):
        super(Encoder, self).__init__()
        self.AttentionCommon = Attention(emb_size, num_attention_heads)
        self.hiddenLayer1 = nn.Linear(emb_size, hidden_size)
        self.hiddenLayer2 = nn.Linear(hidden_size, emb_size)
        self.dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(emb_size)

    def forward(self, dp_emb, dp_mask):
        attention_output, K, V = self.AttentionCommon(dp_emb, dp_mask)

        hiddenLayer1_output = F.relu(self.hiddenLayer1(attention_output))
        hiddenLayer2_output = self.hiddenLayer2(hiddenLayer1_output)
        hiddenLayer3_dropout = self.dropout(hiddenLayer2_output)
        layer_output = self.LayerNorm(hiddenLayer3_dropout + attention_output) #我理解是残差连接

        #输出结果：layer_output表示分割的氨基酸特征
        return layer_output, K, V
