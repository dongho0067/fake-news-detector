import torch.nn as nn
import torch.optim as optim
from data_processor import X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

class FakeNewsLSTM(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim):
    super(FakeNewsLSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_dim*3, output_dim)
    self.dropout = nn.Dropout(0.5)

  def forward(self, input):
    embedded = self.embedding(input)

    lstm_out, (hidden, cell) = self.lstm(embedded)
    output = self.fc(torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1))
    output_dropped = self.dropout(output)
    return self.fc(output_dropped)

# Define hyperparameters
input_size = 10000  # Example vocab size
embedding_dim = 128
hidden_dim = 64
output_dim = 1  # Since it's binary classification (fake/real news)
num_layers = 2
epochs = 10

model = FakeNewsLSTM(input_dim=input_size, output_dim=output_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
def train(model, num_epochs):
  for epoch in range(num_epochs):
    model.train()
    for inputs, labels in zip(X_train_tensor, y_train_tensor): # get from processed data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#evaluation 
def evaluate_model(model):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in zip(X_test_tensor, y_test_tensor):
            outputs = model(inputs.unsqueeze(0))
            predicted = torch.round(torch.sigmoid(outputs)).item()
        
            predictions.append(predicted)
            true_labels.append(labels.item())

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return recall

recall = evaluate_model(model)
print(recall)
threshold = 0.5

if recall >= 0.5:
   print("Real News!")
else:
   print("Fake News")
   
