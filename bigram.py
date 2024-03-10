import os
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

input_file_path = 'D:\\src\\github\\nanoGPT\\data\\shakespeare_char\\input.txt' 
with open(input_file_path, 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
itos = { i:ch for i,ch in enumerate(chars) } #，同时列出数据和数据下标，一般用在 for 循环当中。
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

data = torch.tensor(encode(text),dtype=torch.long)

# create the train and test splits
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.postion_embedding_table = nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and target are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.postion_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the logits for the index in the sequence
            logits, loss = self(idx)
            # focus only the last time step
            logits = logits[:, -1, :] #become (B, C)
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

# helps estimate an arbitrarily accurate loss over either split using many batches
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

# creator a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}:train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    #sample a batch of data
    xb, yb = get_batch(train_data)
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model    
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))