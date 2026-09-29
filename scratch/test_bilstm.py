import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Verify BiLSTM with Attention
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs, mask=None):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs).squeeze(-1) # (batch_size, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1) # (batch_size, seq_len)
        context = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1) # (batch_size, hidden_dim)
        return context, weights

class UrduBiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3, pad_idx=0):
        super(UrduBiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.spatial_dropout = nn.Dropout2d(p=dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        mask = (input_ids != 0)
        embeds = self.embedding(input_ids) # (batch, seq_len, embed_dim)
        
        # Spatial dropout: permute to (batch, embed_dim, seq_len, 1)
        embeds = embeds.permute(0, 2, 1).unsqueeze(-1)
        embeds = self.spatial_dropout(embeds).squeeze(-1).permute(0, 2, 1)
        
        rnn_out, _ = self.lstm(embeds) # (batch, seq_len, hidden_dim * 2)
        context, attn_weights = self.attention(rnn_out, mask=mask)
        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights

# Quick forward pass test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UrduBiLSTMAttention(vocab_size=5000, embed_dim=128, hidden_dim=128).to(device)
dummy_input = torch.randint(1, 5000, (4, 30)).to(device)
dummy_input[:, 25:] = 0 # simulate padding
logits, weights = model(dummy_input)

print("BiLSTM Test Output:")
print("Logits shape:", logits.shape)
print("Attention weights shape:", weights.shape)
print("Logits:", logits)
assert logits.shape == (4,), "Logit shape mismatch!"
assert weights.shape == (4, 30), "Attention shape mismatch!"
print("PyTorch BiLSTM Architecture verified successfully!")
