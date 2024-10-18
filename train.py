import wandb

import torch.nn as nn
import torch
import numpy as np

import torch.nn.functional as F
import re
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from tqdm import tqdm

from rnn import LanguageModelOneHot, LanguageModelWord2Vec
from utils import create_sequences, TextDataset, generate_text, pretty_print_summary
import argparse

# use a parser to get the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='hp.txt', help='path to the data file')
parser.add_argument('--seq_length', type=int, default=50, help='sequence length')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--mode', type=str, default='word2vec', help='onehot or word2vec')
parser.add_argument('--rnn_type', type=str, default='LSTM', help='LSTM or GRU')
parser.add_argument('--if_sampling', type=bool, default=False, help='True for random sampling, False for argmax')
args = parser.parse_args()

# HYPERPARAMETERS
text_file = args.data
seq_length = args.seq_length
batch_size = args.batch_size
mode = args.mode
rnn_type = args.rnn_type
if_sampling = args.if_sampling
hidden_dim = 512  
num_layers = 3
lr = .003
num_epochs = 10

batch_num = 0
log_interval = 100 

pretty_print_summary(args, hidden_dim, num_layers, lr, num_epochs, log_interval, if_sampling)


with open(text_file, 'r', encoding='UTF-8') as f:
    train_data = f.read()

# Split train_data into words
train_data = re.findall(r'\w+|[^\s\w]', train_data)
words_available = sorted(list(set(train_data)))
stoi = {word: i for i, word in enumerate(words_available)}
itos = {i: word for i, word in enumerate(words_available)}
train_data = [stoi[word] for word in train_data]

print("Number of words in training data:", len(train_data))

vocab_size = len(words_available)

train_data = train_data[:10000]

# Initialize wandb with hyperparameters
wandb.init(project="webandtext", config={
    "text_file": text_file,
    "seq_length": seq_length,
    "batch_size": batch_size,
    "mode": mode,
    "rnn_type": rnn_type,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "lr": lr,
    "num_epochs": num_epochs
})

sequences, targets = create_sequences(train_data, seq_length)
dataset = TextDataset(sequences, targets)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if mode == 'onehot':
    model = LanguageModelOneHot(vocab_size, hidden_dim,  num_layers, rnn_type)
elif mode == 'word2vec':
    model = LanguageModelWord2Vec(vocab_size, hidden_dim, num_layers, rnn_type)
else:
    raise ValueError("Invalid mode. Choose 'onehot' or 'word2vec'.")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Move model to device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Using the following backend:", device)

model.to(device)


scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Watch the model with wandb
wandb.watch(model, log='all', log_freq=2)

# Training loop
for epoch in range(num_epochs):
    batch_num += 1

    model.train()
    hidden = None
    total_loss = 0
    
    for inputs, targets in tqdm(train_dataloader):
        hidden = None
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hidden = model(inputs, hidden)
        
        # Detach hidden states to prevent backpropagating through the entire training history
        if isinstance(hidden, tuple):
            hidden = tuple([h.detach() for h in hidden])
        else:
            hidden = hidden.detach()
        
        outputs = outputs.view(-1, vocab_size) 
        targets = targets.view(-1)     
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_num % log_interval == 0:
            wandb.log({'batch_loss': loss.item(), 'batch_num': batch_num, 'epoch': epoch+1})


    avg_loss = total_loss / len(train_dataloader)
    perplexity = np.exp(avg_loss)

    scheduler.step()

    torch.save(model.state_dict(), f'./model_{epoch}.pt')
    
    wandb.log({'epoch': epoch+1, 'avg_loss': avg_loss, 'perplexity': perplexity})

    # Log the model checkpoint as a wandb artifact
    artifact = wandb.Artifact(f'model_epoch_{epoch+1}', type='model', description='Model checkpoint')
    artifact.add_file(f'./model_{epoch}.pt')
    wandb.log_artifact(artifact)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    generated_text = generate_text(model, seed_word="Harry", max_length=50, random_sampling=if_sampling, stoi=stoi, itos=itos, device=device)
    print(generated_text)
