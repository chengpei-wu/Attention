import torch.nn as nn
import math
import torch.nn.functional


class SimpleSelfAttention(nn.Module):
    def __init__(self, show_att=False):
        super(SimpleSelfAttention, self).__init__()
        self.show_att = show_att

    def forward(self, x):
        atten = torch.nn.functional.softmax(torch.bmm(x, x.permute(0, 2, 1)), dim=-1)
        output = torch.bmm(atten, x)
        return (output, atten) if self.show_att else output


class SelfAttention(nn.Module):
    def __init__(self, input_dim, k_dim, v_dim, show_att=False):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(input_dim, k_dim)
        self.K = nn.Linear(input_dim, k_dim)
        self.V = nn.Linear(input_dim, v_dim)
        self.show_att = show_att
        self.normalize_factor = 1 / math.sqrt(k_dim)

    def forward(self, x):
        mat_q = self.Q(x)
        mat_k = self.K(x)
        mat_v = self.V(x)
        atten = torch.nn.functional.softmax(torch.bmm(mat_q, mat_k.permute(0, 2, 1)) * self.normalize_factor,
                                            dim=-1)
        output = torch.bmm(atten, mat_v)
        return (output, atten) if self.show_att else output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, k_dim, v_dim, model_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        assert model_dim % num_heads == 0
        self.Q = nn.Linear(input_dim, k_dim)
        self.K = nn.Linear(input_dim, k_dim)
        self.V = nn.Linear(input_dim, v_dim)
        self.W_Q = nn.Linear(k_dim, model_dim)
        self.W_K = nn.Linear(k_dim, model_dim)
        self.W_V = nn.Linear(v_dim, model_dim)
        self.W_O = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        q, k, v = self.Q(x), self.K(x), self.V(x)

        # note that, the use of W_Q(W_K, W_V) with shape (k_dim, model_dim) is the same as using multiple independent
        # weight matrices(nn.Linear()) with shape (k_dim, model_dim // num_heads)
        multi_q = self.W_Q(q).reshape(-1, x.shape[0], x.shape[1],
                                      self.model_dim // self.num_heads)
        multi_k = self.W_K(k).reshape(-1, x.shape[0], x.shape[1],
                                      self.model_dim // self.num_heads)
        multi_v = self.W_V(v).reshape(-1, x.shape[0], x.shape[1],
                                      self.model_dim // self.num_heads)

        heads_atten = torch.nn.functional.softmax(torch.matmul(multi_q, multi_k.permute(0, 1, 3, 2)), dim=-1)
        concated_heads_atten = torch.matmul(heads_atten, multi_v).reshape(x.shape[0], x.shape[1], -1)

        return self.W_O(concated_heads_atten)
