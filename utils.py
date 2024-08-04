
import os
import json

import random
import numpy as np

import torch
from torch.utils.data import Dataset

from tqdm.notebook import tqdm_notebook as tqdm

class LinregDataset(Dataset):
    def __init__(
        self, 
        n_dims, n_points,
        mean = 0, std = 1, 
        total = 10000, 
        xs = None, ys = None,
        random = True,
        device = 'cpu'
        ):
        
        self.n_dims = n_dims
        self.n_points = n_points
        self.truncate = 0
        
        self.xs = xs # [N, n, d]
        self.ys = ys # [N, n]
        self.length = total if self.xs is None else len(xs)
        self.current = 0
        
        self.mean = mean
        self.std = std
        
        if not random:
            if self.xs is None:
                raise ValueError('You must provide data for random = False.')
            self.sample = self._getdata 
        else:
            self.sample = self._generate
        
        self.device = device
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x, y = self.sample(idx)
        return self.combine(x, y), y
    
    def _generate(self, idx):
        xs = torch.normal(self.mean, self.std, (self.n_points, self.n_dims))
        xs[..., self.n_dims - self.truncate:] = 0
        w_b = torch.normal(0, 1, (self.n_dims, 1))
        w_b[..., self.n_dims - self.truncate:] = 0
        ys = (xs @ w_b).sum(-1)
        return xs.to(self.device), ys.to(self.device)
    
    def _getdata(self, idx):
        xs = self.xs[idx]
        ys = self.ys[idx]
        return xs.to(self.device), ys.to(self.device)
    
    def combine(self, xs, ys):
        n, d = xs.shape
        device = ys.device

        ys_b_wide = torch.cat(
            (
                ys.view(n, 1),
                torch.zeros(n, d-1, device=device),
            ),
            axis = 1,
        )

        zs = torch.stack((xs, ys_b_wide), dim = 1)
        zs = zs.view(2 * n, d)

        return zs

def save(name, model, loss, eval, extr = [], path = '.'):
    os.makedirs(os.path.dirname(path + '/data/'), exist_ok=True)
    os.makedirs(os.path.dirname(path + '/models/'), exist_ok=True)
    
    result = json.dumps({
        'loss': loss,
        'eval': eval,
        'extr': extr
    })
    
    with open(f'{path}/data/{name}.json', 'w') as f:
        f.write(result)
    torch.save(model, f'{path}/models/{name}.pt')
        
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def evaluate(loader, model, b = None):
        
    with torch.no_grad():
        total = 0
        for (x, y) in loader:
            
            preds = model(x[:, :-1], b)
            preds = torch.stack(preds)
            targs = torch.stack([y] * b)
            
            # First by predictions, then by batches
            loss = (targs[:,:,-1] - preds[:,:,-1]).square().mean(dim=0).mean()
            
            total += loss.item() / loader.dataset.n_dims
    return total / len(loader)

def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    b = None,
    steps = None,
    run = None,
    log_every = 1
):
    
    train_loader.dataset.length = steps * train_loader.batch_size
    
    evaluate_every = steps // 5
    
    loss_history = []
    eval_history = []
    
    val_loss = evaluate(test_loader, model, b)
    eval_history.append(val_loss)
    
    if run is not None:
        run.log({'Eval Loss': val_loss}, commit = False, step = 1)
    
    pbar, step = tqdm(range(steps)), 0
    for (x, y) in train_loader:
        optimizer.zero_grad()

        preds = model(x[:, :-1], b)
        preds = torch.stack(preds)
        targs = torch.stack([y] * b)
        
        # First by inputs, then by predictions, then by batches
        loss = (targs - preds).square().mean(dim=2).mean(dim=0).mean()
        loss = loss / train_loader.dataset.n_dims
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        pbar.set_description(f'Train loss: {loss.item():.5f}')
        pbar.update(1)
        
        step += 1
        
        if step % evaluate_every == 0:
            val_loss = evaluate(test_loader, model, b)
            eval_history.append(val_loss)
            if run is not None:
                run.log({'Eval Loss': val_loss}, commit = False, step = step)
        if run is not None and (step % log_every == 0 or step == 1):
            run.log({'Train Loss': loss.item()}, step = step)
    
    return loss_history, eval_history