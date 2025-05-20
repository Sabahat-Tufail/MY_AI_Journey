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
for epoch in range(100):
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
