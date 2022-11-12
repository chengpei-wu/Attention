import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torch.nn.functional


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


class SimpleSelfAttention(nn.Module):
    def __init__(self, show_att=False):
        super(SimpleSelfAttention, self).__init__()
        self.show_att = show_att

    def forward(self, x):
        atten = torch.nn.functional.softmax(torch.bmm(x, x.permute(0, 2, 1)), dim=-1)
        output = torch.bmm(atten, x)
        return (output, atten) if self.show_att else output


x = torch.rand(1, 5, 16)
att = SimpleSelfAttention(show_att=True)
b, atten = att(x)
print(b.size())

# show the attention of input
plt.imshow(atten.detach().numpy().squeeze(), cmap='gray')
plt.colorbar()
plt.show()
