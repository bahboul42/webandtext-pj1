import torch
import torch.nn as nn
from torch.utils.data import Dataset

import re

from tqdm import tqdm

import gensim

###############################################
# I will remove those I don't need at the end #
###############################################

##############################################################################################
# Data preprocessing and loading

# Tokenizing the text using regular expressions
def tokenize_with_re(data):
    tokenized_sentences = [
        re.findall(r'\b[a-zA-Z]+\b', sentence.lower()) for sentence in data if sentence.strip()
    ]
    return tokenized_sentences

##############################################################################################
# Creating sequences
def create_sequences(data, seq_length, stoi):
    sequences = []
    targets = []
    
    for i in tqdm(range(len(data) - seq_length)):
        # Extract sequence and target
        seq = data[i:i+seq_length]
        target = data[i+1:i+seq_length+1]
        
        # Convert each word in the sequence and target to indices using stoi
        seq_indices = [stoi.get(word, 0) for word in seq]
        target_indices = [stoi.get(word, 0) for word in target]
        
        # Only add sequences and targets of the desired length
        if len(seq_indices) == seq_length and len(target_indices) == seq_length:
            sequences.append(seq_indices)
            targets.append(target_indices)
    
    return sequences, targets

##############################################################################################
# Preparing the Dataset
class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

##############################################################################################
# Language Model  
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, embed_type="word2vec", rnn_type='LSTM',  embed_path='wikitext_small_word2vec.model'):
        super(LanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Step 1: Load the Gensim embedding model
        if embed_type == "word2vec":
            self.embedding_model = gensim.models.Word2Vec.load(embed_path)
        elif embed_type == "fasttext":
            self.embedding_model = gensim.models.FastText.load(embed_path)
        else:
            raise ValueError("Invalid embedding type. Please choose 'word2vec' or 'fasttext'.")

        # Get the embedding dimensions from the Gensim model
        embedding_dim = self.embedding_model.wv.vector_size

        # Step 2: Initialize the embedding layer with pretrained weights
        weights = torch.FloatTensor(self.embedding_model.wv.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights)
        
        # Freeze embedding layer
        self.embedding.weight.requires_grad = False

        # Step 3: Initialize RNN (LSTM or GRU)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        else:
            raise ValueError("Invalid RNN type. Choose 'LSTM' or 'GRU'.")
        
        self.fc = nn.Linear(hidden_dim, vocab_size)

    
    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x)
        # embeds shape: (batch_size, seq_length, embedding_dim)
        out, hidden = self.rnn(embeds, hidden)
        # out shape: (batch_size, seq_length, hidden_dim)
        out = out.reshape(-1, self.hidden_dim)
        # out shape: (batch_size * seq_length, hidden_dim)
        out = self.fc(out)
        # out shape: (batch_size * seq_length, vocab_size)
        return out, hidden

##############################################################################################
# Text Generation
def generate_text(model, seed_word, stoi, itos, device, max_length=50, random_sampling=False):
    model.eval()  # Set the model to evaluation mode
    generated_words = [seed_word]  # List to store generated words
    
    # Convert seed word to index
    seed_idx = torch.tensor([[stoi[seed_word]]]).to(device)  # Shape: (1, 1)

    # Initialize hidden state
    hidden = None
    
    # Loop through to generate words
    for _ in range(max_length):
        # Forward pass through the model
        with torch.no_grad():
            output, hidden = model(seed_idx, hidden)
        
        # Get the predicted word (highest probability or sample)
        output = output.squeeze(1)  # Remove the seq_len dimension (now (1, vocab_size))
        if random_sampling:
            # Sample from the output distribution
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.multinomial(probabilities, num_samples=1).item()
        else:
            predicted_idx = torch.argmax(output, dim=1).item()  # Get the index of the word with highest probability
        
        # Convert index back to word
        predicted_word = itos[predicted_idx]
        
        # Append the predicted word to the list
        generated_words.append(predicted_word)
        
        # Set the predicted word as the next input (shape: (1, 1))
        seed_idx = torch.tensor([[predicted_idx]]).to(device)
        
        # Stop if an end-of-sequence token is generated (optional)
        if predicted_word == "<eos>":  # Assuming "<eos>" is the token for end of sentence
            break
    
    return ' '.join(generated_words)

##############################################################################################
# Pretty print of the (hyper)parameters
def pretty_print_summary(args):
    summary = f"""
    -------------------------------------
               Model Summary
    -------------------------------------
    RNN type          : {args.rnn_type}
    Embedding type    : {args.embed_type}
    Sequence Length   : {args.seq_length}
    Batch Size        : {args.batch_size}
    
    ---- Hyperparameters ----
    Hidden Dimension  : {args.hidden_dim}
    Number of Layers  : {args.num_layers}
    Learning Rate     : {args.lr}
    Number of Epochs  : {args.num_epochs}
    -------------------------------------
    """
    print(summary)