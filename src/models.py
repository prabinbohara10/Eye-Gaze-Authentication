import torch.nn as nn
import torch


## LSTM Model
class GazeLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=2):
        super(GazeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  ## Using the last LSTM output
        return out
    
# Model for generating embeddings
class GazeEmbeddingModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, embedding_size=128, num_layers=2):
        super(GazeEmbeddingModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        embedding = self.fc(out[:, -1, :])  # Use last hidden state as embedding
        return embedding