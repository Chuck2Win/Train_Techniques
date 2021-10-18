import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
# 가장 많이 활용하는 lr_scheduler:LambdaLR
scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch:0.95**epoch) # 결국 lr_lambda에 원하는 함수를 넣어주면 됨.
