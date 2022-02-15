import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
# 가장 많이 활용하는 lr_scheduler:LambdaLR
scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch:0.95**epoch) # 결국 lr_lambda에 원하는 함수를 넣어주면 됨.

import transformers
# num_warm_up_steps ; 말그대로 0에서부터 init lr까지 도달하는 step의 수
# num_traini_steps : training step의 수
# num cycles = # of cycles in num training step
scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000, num_training_steps=10000, num_cycles=10)
model = nn.Linear(3,3)
optimizer = transformers.AdamW(model.parameters(),lr=1)
lrs = []
for epoch in range(2):
    for step in range(1000):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
