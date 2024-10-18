import torch.nn as nn
import torch.nn.functional as F

class LanguageModelOneHot(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, rnn_type='LSTM'):
        super(LanguageModelOneHot, self).__init__()
        self.vocab_size = vocab_size
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(vocab_size, hidden_dim, num_layers, batch_first=True) # [batch_size, seq_length, vocab_size] -
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(vocab_size, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose 'LSTM' or 'GRU'.")
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        embeds =  F.one_hot(x, num_classes=self.vocab_size).float()
        out, hidden = self.rnn(embeds, hidden)
        out = out.reshape(-1, out.size(2))
        out = self.fc(out)
        return out, hidden
    
class LanguageModelWord2Vec(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, rnn_type='LSTM'):
        super(LanguageModelWord2Vec, self).__init__()
        self.hidden_dim = hidden_dim
        embedding_dim = 150
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
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

