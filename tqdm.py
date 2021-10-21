# tqdm 
# smart하게 활용할 수 있을 것 같음.

for epoch in range(1,3):
  iter_bar = tqdm(train_dataloader)
  for i in iter_bar:
    iter_bar.set_postfix({"epoch":epoch, "step":step})
