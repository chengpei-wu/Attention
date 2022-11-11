# Implementation of variants of attention

## self attention https://arxiv.org/abs/1706.03762



### math

$$
Attention(Q,K,V)=Softmax(QK^T)V
$$

### usage

```python
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torch.nn.functional


import Attention.Self_Attention as Self_Attention

x = torch.rand(1, 5, 16)
att = Self_Attention(16, 32, 32, show_att=True)
b, atten = att(a)
print(b.size())

# show the attention of input
plt.imshow(atten.detach().numpy().squeeze())
plt.show()
```

