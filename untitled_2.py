import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# Hyperparameters and device setup
context_length = 256 
n_embd = 384 
n_head = 6
n_layer = 6
dropout = 0.0  # Set dropout to zero during inference
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load text and create vocabulary
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping from string to integer and back
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]             # String to list of integers
decode = lambda l: ''.join([itos[i] for i in l])    # List of integers to string

# Define model classes (ensure these match your training code)
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)       # (B, T, head_size)
        q = self.query(x)     # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)     # (B, T, head_size)
        out = wei @ v         # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate along embedding dimension
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple feed-forward network."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand dimension
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Project back to embedding size
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """The main language model."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.block = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        for block in self.block:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Get logits for the last time step
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from distribution
            idx = torch.cat((idx, idx_next), dim=1)  # Append sampled index
        return idx

# Instantiate and load the model
model = BigramLanguageModel()
model.to(device)

# Load the saved weights
model.load_state_dict(torch.load('transformer_weights.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Prepare the initial context (e.g., an empty string or a specific prompt)
initial_text = ''  # You can change this to any starting text
initial_input = torch.tensor(encode(initial_text), dtype=torch.long, device=device).unsqueeze(0)

if initial_input.numel() == 0:
    # If initial_text is empty, start with a tensor of zeros
    initial_input = torch.zeros((1, 1), dtype=torch.long, device=device)

# Set the number of tokens you want to generate
max_new_tokens = 1000

# Measure loading time
start_time = time.time()
model.load_state_dict(torch.load('transformer_weights.pth', map_location=device))
model.eval()
load_time = time.time() - start_time
print(f"Model load time: {load_time:.4f} seconds")

# Measure inference time
initial_input = torch.zeros((1, 1), dtype=torch.long, device=device)
max_new_tokens = 1000

start_time = time.time()
with torch.no_grad():
    output = model.generate(initial_input, max_new_tokens=max_new_tokens)
inference_time = time.time() - start_time

# Decode the output
generated_text = decode(output[0].tolist())
print(generated_text)
print(f"Inference time: {inference_time:.4f} seconds")
