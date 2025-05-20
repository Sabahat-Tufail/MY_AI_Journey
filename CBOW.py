import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample data
text = "the quick brown fox jumps over the lazy dog".split()

# Create vocabulary
vocab = set(text)
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}
vocab_size = len(vocab)

# Generate CBOW training pairs
def generate_cbow_pairs(text, window_size=2):
    pairs = []
    for i in range(window_size, len(text) - window_size):
        context = [text[i - j - 1] for j in range(window_size)] + \
                  [text[i + j + 1] for j in range(window_size)]
        target = text[i]
        pairs.append((context, target))
    return pairs

cbow_pairs = generate_cbow_pairs(text)

# Convert to tensors
def one_hot(index):
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec

X = []
Y = []

for context, target in cbow_pairs:
    context_vec = np.sum([one_hot(word2idx[w]) for w in context], axis=0)
    X.append(context_vec)
    Y.append(word2idx[target])

X = torch.Tensor(X)
Y = torch.LongTensor(Y)

# Define CBOW Model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Linear(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.output(x)
        return x

model = CBOW(vocab_size, 10)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(400):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, Y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test prediction
with torch.no_grad():
    for i in range(5):
        context, target = cbow_pairs[i]
        input_vec = torch.Tensor(np.sum([one_hot(word2idx[w]) for w in context], axis=0))
        output = model(input_vec)
        predicted_idx = torch.argmax(output).item()
        print(f"Context: {context} -> Predicted: {idx2word[predicted_idx]}, Actual: {target}")
'''
#task 2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample corpus
text = "the quick brown fox jumps over the lazy dog".split()

# Create vocabulary
vocab = set(text)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

# Generate CBOW pairs with window size 2
def generate_cbow_pairs(text, window_size=2):
    pairs = []
    for i in range(window_size, len(text) - window_size):
        context = [text[i - j - 1] for j in range(window_size)] + [text[i + j + 1] for j in range(window_size)]
        target = text[i]
        pairs.append((context, target))
    return pairs

cbow_pairs = generate_cbow_pairs(text)

# One-hot encoding function
def one_hot(idx):
    vec = np.zeros(vocab_size)
    vec[idx] = 1
    return vec

# Prepare input and target tensors
X = []
Y = []
for context, target in cbow_pairs:
    # Sum or average one-hot vectors for context
    context_vec = np.mean([one_hot(word2idx[w]) for w in context], axis=0)  # averaging here
    X.append(context_vec)
    Y.append(word2idx[target])

X = torch.FloatTensor(X)
Y = torch.LongTensor(Y)

# Define CBOW Model with one hidden layer
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        # First linear layer = embedding layer
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.activation = nn.ReLU()
        # Second linear layer = output layer
        self.linear2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Instantiate model, loss, optimizer
embedding_dim = 10
model = CBOW(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Training loop
epochs = 400
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            output = model(X)
            predicted = torch.argmax(output, dim=1)
            correct = (predicted == Y).sum().item()
            total = Y.size(0)
            accuracy = correct / total * 100
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

# Test some predictions
with torch.no_grad():
    for i in range(5):
        context, target = cbow_pairs[i]
        input_vec = torch.FloatTensor(np.mean([one_hot(word2idx[w]) for w in context], axis=0))
        output = model(input_vec)
        predicted_idx = torch.argmax(output).item()
        print(f"Context: {context} -> Predicted: {idx2word[predicted_idx]}, Actual: {target}")'''