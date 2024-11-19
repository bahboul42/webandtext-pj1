import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torcheval.metrics import WordErrorRate, Perplexity

import wandb

from tqdm import tqdm

import gensim

import argparse

from datasets import load_dataset

from utils import tokenize_with_re, create_sequences, TextDataset, LanguageModel, pretty_print_summary


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print('Using device:', device)

# wandb.login()

def training_loop(model, dataloader, itos, vocab_size, hidden_dim=256, num_layers=3,
                  rnn_type="lstm", learning_rate=0.0003, num_epochs=10, seq_length=50):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize WER and Perplexity for both train and validation
    wer = WordErrorRate()
    train_perplexity = Perplexity()

    if torch.cuda.is_available():
        wer = wer.to(device)
        train_perplexity = train_perplexity.to(device)

    train_losses = {'crossentropy': [], 'wer': [], 'perplexity': []}

    # Move model to device (GPU or CPU)
    model.to(device)  

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        hidden = None

        # Initialize metrics for the epoch
        wer.reset()
        train_perplexity.reset()

        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
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
            targets_buffer = targets
            targets = targets.view(-1)

            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Calculate WER
            # For WER and Perplexity, keep the original targets shape
            preds = torch.argmax(outputs, dim=1)

            spreds = [itos[p.item()] for p in preds]
            stargets = [itos[t.item()] for t in targets]  # You might want to keep the same targets as for loss
            wer.update(spreds, stargets)
            
            # Calculate perplexity using the original (2D) targets
            if torch.cuda.is_available():
                train_perplexity.update(outputs.view(outputs.size(0) // seq_length, seq_length, -1), targets_buffer)
            else:
                train_perplexity.update(outputs.view(outputs.size(0) // seq_length, seq_length, -1).cpu(), targets_buffer.cpu())  # Pass original target shape
        
        # Calculate training metrics
        avg_train_loss = total_loss / len(dataloader)
        avg_train_wer = wer.compute().item()
        avg_train_perplexity = train_perplexity.compute().item()
        
        train_losses['crossentropy'].append(avg_train_loss)
        train_losses['wer'].append(avg_train_wer)
        train_losses['perplexity'].append(avg_train_perplexity)

        # Save model
        torch.save(model.state_dict(), f'model_weights/model_{rnn_type}_{embedding_type}_{hidden_dim}hidden_{num_layers}layers_{learning_rate}lr_{epoch+1}epoch.pt')

        # Log the metrics to W&B
        wandb.log({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'train_WER': avg_train_wer,
            'train_perplexity': avg_train_perplexity,
        })

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training WER: {avg_train_wer:.4f}, Training Perplexity: {avg_train_perplexity:.4f}')

    return train_losses

if __name__ == "__main__":
    # Still need to add the arguments in this parser!
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_type", type=str, default='LSTM', help='LSTM or GRU') # RNN type
    parser.add_argument("--embed_type", type=str, default='word2vec') # Type of embedding
    parser.add_argument("--batch_size", type=int, default=32, help='batch size') # Batch size
    parser.add_argument("--seq_length", type=int, default=50, help='sequence length') # Sequence length
    parser.add_argument("--hidden_dim", type=int, default=256, help='hidden dimension') # Hidden dimension
    parser.add_argument("--num_layers", type=int, default=3, help='number of layers') # Number of layers
    parser.add_argument("--lr", type=float, default=0.0003, help='learning rate') # Learning rate
    parser.add_argument("--num_epochs", type=int, default=10) # Number of epochs
    parser.add_argument("--run_nbr", type=int, default=1, help='run number') # Run number
    args = parser.parse_args()

    # (Hyper)parameters
    batch_size = args.batch_size
    seq_length = args.seq_length
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    learning_rate = args.lr
    num_epochs = args.num_epochs
    rnn_type = args.rnn_type # Choose 'LSTM' or 'GRU'
    embedding_type = args.embed_type # 'word2vec' or 'fasttext'

    run_name = f"{embedding_type}_{rnn_type}_{args.run_nbr}"
    wandb.init(project='webandtext-pj1',
            entity='andreascoco',
            name=run_name,
            tags = [embedding_type, rnn_type],
            group = embedding_type,
            config={
                'num_layers': num_layers,
                'hidden_dim': hidden_dim,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            })

    # Pretty print of the (hyper)parameters
    pretty_print_summary(args)

    train_data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    train = train_data['train']['text']

    # Tokenize each dataset split
    train_tokenized = tokenize_with_re(train)
    train_tokenized = train_tokenized[:len(train_tokenized)//2]  # Only use half of the training data

    flat_train = [item for sublist in train_tokenized for item in sublist]

    # Getting embedding path
    if embedding_type == "word2vec":
        embed_path = 'wikitext_small_word2vec.model'
    elif embedding_type == "fasttext":
        embed_path = 'wikitext_small_fasttext.model'

    # Prepare the data:
    if embedding_type == "word2vec":
        embedding_model = gensim.models.Word2Vec.load(embed_path)
    elif embedding_type == "fasttext":
        embedding_model = gensim.models.FastText.load(embed_path)
    else:
        raise ValueError("Invalid embedding type. Please choose 'word2vec' or 'fasttext'.")
    
    stoi = {word: idx for idx, word in enumerate(embedding_model.wv.index_to_key)}
    itos = {idx: word for idx, word in enumerate(embedding_model.wv.index_to_key)}

    # Creating the sequences
    sequences, targets = create_sequences(flat_train, seq_length, stoi)

    # Creating Dataset and DataLoader
    dataset = TextDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Preparing for the training loop
    vocab_size = len(embedding_model.wv)
    print(f"Total number of words: {len(flat_train)}, vocabulary size: {vocab_size}")

    # Initialize the model
    model = LanguageModel(vocab_size, hidden_dim, num_layers, embedding_type, rnn_type, embed_path)

    # Train the model
    train_losses = training_loop(model, dataloader, itos, vocab_size, hidden_dim, num_layers, rnn_type, learning_rate, num_epochs, seq_length)


