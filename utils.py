import torch
from torch.utils.data import Dataset

# Dataset

# create sequences from the text of specific length
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+1:i+seq_length+1]
        if len(target) < seq_length or len(seq) < seq_length:
            break
        sequences.append(seq)
        targets.append(target)
    return sequences, targets

# dataset class that returns the sequences and targets
class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

# Generate text from a model and a seed word
def generate_text(model, seed_word, max_length=50, random_sampling=False, stoi=None, itos=None, device='cpu'):
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
        
        predicted_word = itos[predicted_idx]
        generated_words.append(predicted_word)
        seed_idx = torch.tensor([[predicted_idx]]).to(device)
        
        if predicted_word == "<eos>":  
            break
    
    return ' '.join(generated_words)

def compute_perplexity(loss):
    return torch.exp(loss)

# Perplexity
# model.eval()
# with torch.no_grad():
#     total_loss = 0
#     for batch in valid_iterator:
#         inputs, targets = batch.text[:, :-1], batch.text[:, 1:]
#         outputs, hidden = model(inputs, None)
#         loss = criterion(outputs, targets.reshape(-1))
#         total_loss += loss.item()

# perplexity = compute_perplexity(total_loss / len(valid_iterator))
# print(f'Validation Perplexity: {perplexity}')
