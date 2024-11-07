import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameter 
batch_size = 64 # how many independent sequences will be process in parallel
context_length = 256 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # number of embedding dimensions 
n_head = 6
n_layer = 6
dropout = 0.2
# --------------

torch.manual_seed(420)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All the unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mapping from string to integer and back
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]             # string -> integer list
decode = lambda l: ''.join([itos[i] for i in l])    # integer list -> string 

# Train test split
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate small batch of data for x and y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1]for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# everything in the following function we won't call .backward() on, more efficient in memory use since don't need to store intermediate variable to process 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinity")
        wei = q @ k.transpose(-2, -1) * C ** -0.5       # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)                    # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        v = self.value(x)       # (B, T, C)
        out = wei @ v           # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
         
    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # multipliyer of 4 from attention paper s
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer back into residual path way 
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimensions, n_head: the number of heads we'd like 
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # layer norm is applied before the attention, in contrast to after the self attention in attention paper
        x = x + self.ffwd(self.ln2(x))
        return x

# simple bigram model 
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly read off the logits for the next token from a look up table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.block = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm   
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)     # (B,T,C), score for next token in a sequence
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb     # (B, T, C)
        #x = self.sa_head(x)         # apply one head of attention. (B, T, C); gather data from all tokens
        #x = self.ffwd(x)            # (B, T, C); give time to the network to 'think' about what they found from looking at each other 
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        if targets is None: 
            loss = None 
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indicies in the current context
        for _ in range(max_new_tokens):
            # crop idx to the lass context_length tokens
            idx_cond = idx[:, -context_length:]
            # get the prediction 
            logits, loss = self(idx_cond)
            # focus only on the last time step 
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim = -1) # (B, C)
            # Sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx 
    
model = BigramLanguageModel()
m = model.to(device)

# pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters() ,lr = learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data 
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate
context = torch.zeros([1, 1], dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate the model and print the parameter count
print(f'Total Parameters: {count_parameters(m)}')

# Saving model weights for fine-tuning or quantization experiments
torch.save(m.state_dict(), 'transformer_weights.pth')