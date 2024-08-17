
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_length):
        super(SinusoidalPositionEmbedding, self).__init__()

        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(max_length).view(-1, 1)
        wk = 1 / math.log(10000.0) ** (torch.arange(0, hidden_dim, 2) / hidden_dim)

        pe[:, 0::2] = torch.sin(position * wk)
        pe[:, 1::2] = torch.cos(position * wk)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.attn_heads == 0
        
        self.attn = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.hidden_dim = config.hidden_dim
        self.attn_heads = config.attn_heads
    
    def forward(self, x, mask = None):
        B, T, C = x.size()
        
        qkv = self.attn(x) # [B, T, 3 * hidden_dim]
        q, k, v = qkv.split(self.hidden_dim, dim=2)
        # We transform q, k, v to have shape [B, num heads, T, head size]:
        q = q.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2)
        k = k.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2) 
        v = v.view(B, T, self.attn_heads, C // self.attn_heads).transpose(1, 2)
        
        # Logits will have shape [B, num heads, T, T]
        logits = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))
        if mask is not None:
            logits = logits.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        scores = logits.softmax(dim=-1) 
        
        out = scores @ v # [B, num heads, T, head size]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.linear(out)
        
        return out, scores

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.mlp_hidden)
        self.gelu = config.activation()
        self.linear2 = nn.Linear(config.mlp_hidden, config.hidden_dim)
    
    def forward(self, x):
        x = self.linear2(self.gelu(self.linear1(x)))
        return x
    
class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)
        
    def forward(self, x, mask = None):
        attn, scores = self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x + attn))
        return x, scores
    
class LoopedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.emb = nn.Linear(config.n_dims, config.hidden_dim)
        self.pe = nn.Embedding(config.context, config.hidden_dim)
        self.layers = nn.Sequential(*[
            Layer(config) for i in range(config.num_layers)
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


class BaseLoopedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.emb = nn.Linear(config.n_dims, config.hidden_dim)
        self.layers = nn.Sequential(*[
            Layer(config) for i in range(config.num_layers)
        ])
        self.out = nn.Linear(config.hidden_dim, config.n_dims)
        mask = torch.tril(torch.ones(self.config.context, self.config.context))
        mask = mask.view(1, 1, self.config.context, self.config.context)
        self.register_buffer('mask', mask)
        
    def forward(self, x, b = 1):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        x = self.emb(x)
        output = torch.zeros_like(x)
        
        pred_list = []
        for i in range(b):
            output = output + x
            for layer in self.layers:
                output, scores = layer(output, self.mask)
            prediction = self.out(output)[:, ::2, 0]
            pred_list.append(prediction)
            
        return pred_list