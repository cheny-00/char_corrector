import torch
from trainer.trainer import BaseTrainer

class LMTrainer(BaseTrainer):
    
    def __init__(self, optim, seq_len, lr, batch_size, emb_dim, train_data, dev_data, tb_writer, logger, eval_interval, scheduler=None, ckp_save_path=None, train_step=0):
        super().__init__(optim, seq_len, lr, batch_size, emb_dim, train_data, dev_data, tb_writer, logger, eval_interval, scheduler, ckp_save_path, train_step)
    
    def train_process(self, model, data, **kwargs):
        
        self.optim.zero_grad()
    
    def eval_model(self, model, **kwargs):
        
        model.eval()
        
        dev_total_loss = 0
        
        with torch.no_grad():
            for data in self.dev_data:
                _, dev_loss = model(data)
                dev_total_loss += dev_loss
        return dev_total_loss / len(self.dev_data)