import torch
from trainer.trainer import BaseTrainer

class LMTrainer(BaseTrainer):
    
    def __init__(self, optim, seq_len, batch_size, emb_dim, train_data, dev_data, tb_writer, logger, eval_interval, scheduler=None, ckp_save_path=None, train_step=0):
        super().__init__(optim, seq_len, batch_size, emb_dim, train_data, dev_data, tb_writer, logger, eval_interval, scheduler, ckp_save_path, train_step)
    
    def train_process(self, model, data, **kwargs):
        criterion = kwargs['criterion']
        hidden = model.init_hidden()
        total_loss = 0
        n_data = 0
        for x, y in self.get_batch(data):
            
            self.optim.zero_grad()
            hidden = repackage_hidden(hidden)
            
            logit, hidden = model(x, hidden)
            loss = criterion(logit, y)
            
            total_loss += loss.float().item()
            n_data += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            
            """
            # SGD
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            """
            
            self.optim.step()
        mean_loss = total_loss / n_data
        return mean_loss
    
    def eval_model(self, model, **kwargs):
        
        model.eval()
        
        dev_total_loss = 0
        
        with torch.no_grad():
            for data in self.dev_data:
                _, dev_loss = model(data)
                dev_total_loss += dev_loss
        return dev_total_loss / len(self.dev_data)
    
    def get_batch(self, data):
        """
        Subdivides the source into chunks of length bptt.
        The chunks are along dimension 0 (length of each row is n_seq).
        """
        seq_len = self.seq_len
        for i in range(0, len(data['x']), seq_len-1):
            # seq_len = min(self.seq_len, len(source)-1-i)
            x = data[x][:, i:i+seq_len]
            y = data[x][:, i+1:i+1+seq_len].view(-1)
            yield x, y


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)