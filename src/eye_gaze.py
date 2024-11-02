import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class GazeDataset(Dataset):
    def __init__(self, csv_file, sequence_length=10):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.data['Name'] = self.data['Name'].astype('category')
        self.data['Label'] = self.data['Name'].cat.codes ##here, Converting  names to numeric labels
        self.labels = self.data['Label'].unique()
        
        #Grouping data by person
        self.groups = self.data.groupby('Label')

        ## Prepare sequences
        self.sequences = []
        self.sequence_labels = []
        
        for label, group in self.groups:
            coords = group[['X', 'Y']].values
            for i in range(len(coords) - self.sequence_length + 1):
                sequence = coords[i:i + self.sequence_length]
                self.sequences.append(sequence)
                self.sequence_labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.sequence_labels[idx], dtype=torch.long)
        return sequence, label

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
        out = self.fc(out[:, -1, :])  # Use the last LSTM output
        return out

### Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# # Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')



#Setting Hyperparameters 
sequence_length = 10
batch_size = 16
num_epochs = 25
learning_rate = 0.001

#Loadinggg data
csv_file = 'data/point_data/gaze_data_combined.csv'
dataset = GazeDataset(csv_file, sequence_length=sequence_length)

#Splitting hthe into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Creating Model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GazeLSTM(num_classes=len(dataset.labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Training the model the model
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

### Evaluation of the model
evaluate_model(model, test_loader)
