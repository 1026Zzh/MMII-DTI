import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SearchAttention(nn.Module):
    def __init__(self,emb_size, num_attention_heads):
        super(SearchAttention,self).__init__()
        if emb_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(emb_size / num_attention_heads) #每个注意力头的输入维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(emb_size, self.all_head_size) #每个线性变换层的输入维度是 emb_size，输出维度是 self.all_head_size
        self.key = nn.Linear(emb_size, self.all_head_size)
        self.value = nn.Linear(emb_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) #张量维度的交换

    def forward(self, dp_emb, dp_mask, K, V, M):
        mixed_query_layer = self.query(dp_emb)

        query_layer = self.transpose_for_scores(mixed_query_layer) #调整查询的张量维度
        key_layer = K
        value_layer = V

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores - M #实现对某些位置的屏蔽或掩码，以便模型关注或忽略特定位置的信息。

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = (attention_probs.transpose(-1, -2) * dp_mask).transpose(-1, -2)

        target_score = torch.sum(torch.amax(attention_probs, dim=2), dim=1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #创建一个新的形状，通过移除最后两个维度，然后添加一个新的维度
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, target_score


class Decoder2D(nn.Module):
    def __init__(self, emb_size, num_attention_heads, hidden_size):
        super(Decoder2D, self).__init__()
        self.searchAttention = SearchAttention(emb_size, num_attention_heads) #实例化SearchAttention模型
        self.attentonLayer = nn.Linear(emb_size, emb_size) #输入：emb_size； 输出：emb_size
        self.dropout1 = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(emb_size) #输入的形状与输出的形状是一致的

        self.hiddenLayer1 = nn.Linear(emb_size, hidden_size) #输入：emb_size； 输出：hidden_size
        self.hiddenLayer2 = nn.Linear(hidden_size, emb_size) #输入：hidden_size； 输出：emb_size
        self.dropout2 = nn.Dropout(0.1)

    def forward(self,dp_emb, dp_mask, K, V, M):
        dp_mask = dp_mask.unsqueeze(1).unsqueeze(2) #使其变为一个四维张量
        M = M.unsqueeze(1).unsqueeze(2) #使其变为一个四 维张量
        M = (1.0 - M) * 100000.0
        searchAttention_output, target_score = self.searchAttention(dp_emb, dp_mask, K, V, M)
        attentionLayer_output = self.attentonLayer(searchAttention_output)
        attentionLayer_dropout = self.dropout1(attentionLayer_output)
        attention_output = self.LayerNorm(attentionLayer_dropout + dp_emb)

        hiddenLayer1_output = F.relu(self.hiddenLayer1(attention_output))
        hiddenLayer2_output = self.hiddenLayer2(hiddenLayer1_output)
        hiddenLayer3_dropout = self.dropout2(hiddenLayer2_output)

        layer_output = self.LayerNorm(hiddenLayer3_dropout + attention_output)
        # 输出结果：layer_output表示药物二维子结构的特征向量；
        #         target_score表示药物的二维子结构注意力分布（用作损失函数）
        return layer_output, target_score
