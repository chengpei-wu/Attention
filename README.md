# Implementation of variants of attention

## self attention



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


import Attention.SelfAttention as SelfAttention

x = torch.rand(1, 5, 16)
att = SelfAttention(16, 32, 32, show_att=True)
b, atten = att(x)
print(b.size())

# show the attention of input
plt.imshow(atten.detach().numpy().squeeze())
plt.show()
```

