
import torch
import torch.nn as nn
import math
    
def normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-6)

class Kernel(nn.Module):
    def __init__(self, config):
        super(Kernel, self).__init__()

        self.gamma = nn.Parameter(torch.rand(config.hidden_dim))
        self.beta = nn.Parameter(torch.rand(config.hidden_dim))
        
        self.normalize = config.normalize_kernel
        
    def forward(self, x):
        if self.normalize:
            x = normalize(x)
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
        self.causal = config.mask_type == 'causal'
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.attn(x)
        q, k, v = qkv.split(self.hidden_dim, dim=2)
        
        kerq = self.phiq(q)
        kerk = self.phik(k)
        
        kerq = kerq.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2)
        kerk = kerk.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2) 
        v = v.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2)
        
        # Linear attention
        
        if self.causal:
            kv = torch.einsum('bhtf,bhtg->bhtfg', kerk, v).cumsum(dim=2)
            out = torch.einsum('bhtf,bhtfg->bhtg', kerq, kv)
        else:
            # This method should be rewritten for multi-headed attention
            # (should it?)
            
            # numerator = kerq @ (kerk.transpose(-2, -1) @ v)
            # denominator = kerq @ kerk.sum(dim=-2).transpose(-2, -1)
            # out = numerator / denominator
            
            raise NotImplementedError('This method is not implemented yet.')
        
        Z = 1 / (torch.einsum("bhtf,bhtf->bht", kerq, kerk.cumsum(2)) + 1e-6)
        out = out * Z[:, :, :, None]
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
    
class ReBasedLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attn = ReBasedAttention(config)
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)
        
        self.normalize = config.normalize_input
        
    def forward(self, x):
        
        if self.normalize:
            attn = self.attn(normalize(x))
        else:
            attn = self.attn(x)
            
        x = x + self.mlp(self.ln(x + attn))
        return x
    
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
        
    def forward(self, x, b = 1):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        x = self.emb(x)
        x = x + self.pe(torch.arange(x.size(1), device=x.device))
        output = torch.zeros_like(x)
        
        pred_list = []
        for i in range(b):
            output = output + x # Input Injection
            for layer in self.layers:
                output = layer(output)
            prediction = self.out(output)[:, ::2, 0]
            pred_list.append(prediction)
            
        return pred_list