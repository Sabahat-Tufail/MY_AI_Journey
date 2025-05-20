'''Simple RNN Cell from Scratch (Using NumPy
import numpy as np
inputs = [1, 2, 3]

# Initialize weights and bias
W = 0.5     # input to hidden
U = 0.8     # hidden to hidden
b = 0.0
h = 0       # initial hidden state


def tanh(x):
    return np.tanh(x)


for t, x in enumerate(inputs):
    h = tanh(W * x + U * h + b)
    print(f"Time step {t+1}: h = {h:.4f}")

# Task 1: Implement a simple RNN model using PyTorch to predict the next digit in a sequence of digits (0-9).

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h
    
    def predict(self, x):
        h = torch.zeros(1, x.size(0), hidden_size)  # batch size from input
        out, _ = self.forward(x, h)
        out = out.squeeze(0)
        probs = F.softmax(out, dim=1)
        pred_indices = torch.argmax(probs, dim=1)
        return pred_indices

# Define characters as digits 0-9
chars = ['0','1','2','3','4','5','6','7','8','9']

input_size = len(chars)    # 10
hidden_size = 15
output_size = len(chars)   # 10

model = CharRNN(input_size, hidden_size, output_size)

# One-hot encoding of digits '0', '1', '2' for example input sequence
x = torch.tensor([
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # '0'
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # '1'
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # '2'
    ]
], dtype=torch.float32)

# Target output: next digits, say '1', '2', '3' indices = 1,2,3
target = torch.tensor([1, 2, 3], dtype=torch.long).unsqueeze(0)  # batch=1, seq_len=3

# Training parameters
epochs = 300
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    h0 = torch.zeros(1, 1, hidden_size)  # initial hidden state (num_layers=1, batch=1)
    output, _ = model(x, h0)             # output shape: (1, seq_len, output_size)
    output = output.squeeze(0)            # shape: (seq_len, output_size)

    loss = criterion(output, target.squeeze(0))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Predict using model.predict (returns predicted indices)
predicted_indices = model.predict(x)

print("\nPredicted next digits:")
for idx in predicted_indices:
    print(chars[idx.item()])
# The above code is a simple RNN model that predicts the next digit in a sequence of digits.'''

# Task 2 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate dummy data
X=np.array([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
y=np.array([[3], [4], [5], [6]])

# Build the model
model = Sequential([SimpleRNN(10, activation='tanh', input_shape=(3, 1)), Dense()])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Predict
test_input = np.array([[[4], [5], [6]]])
print("Predicted next value:", model.predict(test_input))