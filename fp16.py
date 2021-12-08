import torch.nn.utils as torch_utils

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

model = 모델 초기화
optimizer = 옵티마이저(model.parameters(), ...)

# 생략 ...

scaler = GradScaler()

for step, batch in enumerate(tqdm(train_data_loader, desc="Train", ncols=80)):

    model.zero_grad()

    with autocast():
        logits = model(
            input_ids=ids,
            token_type_ids=token_type_ids,
            attention_mask=mask
        )

        loss = torch.nn.CrossEntropyLoss(weight=ce_weights)(
            logits, rel_type
        )  # logits => (batch * class_num), rel_type => (batch * 1)
        loss /= iters_to_accumulate

    scaler.scale(loss).backward()

    if (step + 1) % iters_to_accumulate == 0:
        torch_utils.clip_grad_norm_(
            model.parameters(),
            1e8,
        )

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
