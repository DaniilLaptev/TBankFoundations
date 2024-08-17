

import torch
import torch.nn as nn
import math

class Kernel(nn.Module):
    def __init__(self, config):
        super(Kernel, self).__init__()

        self.gamma = nn.Parameter(torch.rand(config.hidden_dim))
        self.beta = nn.Parameter(torch.rand(config.hidden_dim))
        
    def forward(self, x):
        return (self.gamma * x + self.beta).square()

class ReBasedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.attn_heads == 0
        
        self.attn = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.phiq = Kernel(config)
        self.phik = Kernel(config)
        
        self.hidden_dim = config.hidden_dim
        self.attn_heads = config.attn_heads
    
    def forward(self, x, mask = None):
        B, T, C = x.size()
        
        qkv = self.attn(x)
        q, k, v = qkv.split(self.hidden_dim, dim=2)
        
        kerq = self.phiq(q)
        kerk = self.phik(k)
        
        numerator = kerq @ (kerk.transpose(-2, -1) @ v)
        denominator = kerq @ kerk.sum(dim=-2).transpose(-2, -1)
        
        out = numerator / denominator
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.linear(out)
        
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.mlp_hidden)
        self.gelu = config.activation()
        self.linear2 = nn.Linear(config.mlp_hidden, config.hidden_dim)
    
    def forward(self, x):
        x = self.linear2(self.gelu(self.linear1(x)))
        return x
    
def normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-6)
    
class ReBasedLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = ReBasedAttention(config)
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)
        
    def forward(self, x, mask = None):
        attn, scores = self.attn(normalize(x), mask)
        x = x + self.mlp(self.ln2(x + attn))
        return x, scores
    
class LoopedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.emb = nn.Linear(config.n_dims, config.hidden_dim)
        self.pe = nn.Embedding(config.context, config.hidden_dim)
        self.layers = nn.Sequential(*[
            ReBasedLayer(config) for i in range(config.num_layers)
        ])
        self.out = nn.Linear(config.hidden_dim, config.n_dims)
         
    def _get_mask(self, config, input_dim, device):
        if config.mask_type == 'causal':
            mask = torch.tril(torch.ones(input_dim, input_dim))
            return mask.view(1, 1, input_dim, input_dim).to(device)
        else:
            raise NotImplementedError(f'Mask type \'{config.mask_type}\' is not implemented.')
        
    def forward(self, x, b = 1):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        mask = self._get_mask(self.config, x.size(1), x.device)
        
        x = self.emb(x)
        x = x + self.pe(torch.arange(x.size(1), device=x.device))
        output = torch.zeros_like(x)
        
        pred_list = []
        for i in range(b):
            output = output + x # Input Injection
            for layer in self.layers:
                output, scores = layer(output, mask)
            prediction = self.out(output)[:, ::2, 0]
            pred_list.append(prediction)
            
        return pred_list