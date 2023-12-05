import torch.nn as nn
import torch.optim as optim
from Detector_AI import NewsData
import torch

class FakeNewsDetector(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim):
    super(FakeNewsDetector, self).__init__()
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, input):
    embedded = self.embedding(input)

    output, (hidden, cell) = self.lstm(embedded)
    return self.fc(hidden.squeeze(0))


optimizer = optim.Adam()
criterion = nn.BCEWithLogitsLoss

#training loop
def train(model, num_epochs):
  for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader: # get from processed data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#evaluation 
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
