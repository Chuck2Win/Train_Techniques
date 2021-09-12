# Early stopping 
# validation loss가 증가하기 시작할 때 train을 멈추는 것.

# hyperpara - patience
import torch

class EarlyStopping(object):
    def __init__(self, patience, save_dir, min_difference=0.):
        self.patience = patience
        self.min_difference = min_difference
        self.min_loss = float(inf)
        self.min_model = None
        self.min_count = 0
        self.timetobreak = False
        self.save_dir = save_dir
    def check(self, model, val_loss):
        if self.min_loss-val_loss>self.min_difference:
            self.min_loss = val_loss
            self.min_count = 0
            self.min_model = model
        elif self.min_loss-val_loss<self.min_difference:
            self.min_count+=1
            if self.min_count>=self.patience:
                self.timetobreak=True
                torch.save(model, self.save_dir)
        
