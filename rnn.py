import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, rnn_type='LSTM'):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose 'LSTM' or 'GRU'.")
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        embeds = self.embedding(x)
        out, hidden = self.rnn(embeds, hidden)
        out = out.reshape(-1, out.size(2))
        out = self.fc(out)
        return out, hidden
