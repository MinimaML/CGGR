import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        # Simple embedding and pos encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, d_model)) # Approx max len 512
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # x: (B, S)
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x, src_mask=mask)
            
        logits = self.fc_out(x)
        return logits, x
