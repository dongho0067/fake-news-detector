import torch.nn as nn
from Detector_AI import NewsData

class FakeNewsDetector(nn.Module):
  def __init__(self, input_dim):
    super(FakeNewsDetector, self).__init__()
    self.fc = nn.Linear(input_dim, 1)
    self.sigmoid = nn.Sigmoid()
    self.loss_f = nn.BCELoss()
    self.optimizer = nn.optim.Adam()
    self.news_data = NewsData()

  def forward(self, x):
    x = self.fc(x)
    return self.sigmoid(x)


  def train(self, model):
    epochs = 10
    for epoch in range(epochs):
      self.optimizer.zero_grad()
      outputs = model() #input into model
      loss = self.loss_f(outputs, _) # apply loss function on outputs and labels
      loss.backward()
      self.optimizer.step()
  
  def evaluate(self):
    pass