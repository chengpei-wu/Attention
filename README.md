# Implementation of variants of attention

## Self Attention



### math

$$
Attention(Q,K,V)=Softmax(QK^T)V
$$

### usage

```python
import torch.nn as nn
import matplotlib.pyplot as plt


import Attention.SelfAttention as SelfAttention

x = torch.rand(1, 5, 16)
attention = SelfAttention(16, 64, 64, show_att=True)
y, atten = attention(x)
print(y.size())
# torch.Size([1, 5, 64])

# show the attention of input
plt.imshow(atten.detach().numpy().squeeze())
plt.show()
```

## Multi Self Attention

### math

$$
\begin{align}
MultiHead(Q,K,V)=Concate(head_1,\dots,head_n)W^O \\
where\ head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)
\end{align}
$$

### usage

```python
import torch.nn as nn
import torch.nn.functional


import Attention.MultiHeadSelfAttention as MultiHeadSelfAttention

x = torch.rand(1, 5, 16)
attention = MultiHeadSelfAttention(16, 64, 64, 8 * 64, 8)
y = attention(x)
print(y.size())
# torch.Size([1, 5, 512])
```