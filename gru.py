import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

# Load and preprocess
tokenizer = get_tokenizer('basic_english')
train_iter = IMDB(split='train')

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def encode(text):
    return vocab(tokenizer(text))

def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(1 if label == 'pos' else 0)
        texts.append(torch.tensor(encode(text), dtype=torch.int64))
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return torch.tensor(labels), texts

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(len(vocab), 64, 128, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_iter = IMDB(split='train')
    dataloader = DataLoader(list(train_iter), batch_size=16, shuffle=True, collate_fn=collate_batch)

    for epoch in range(3):
        total_loss = 0
        model.train()
        for labels, texts in dataloader:
            labels, texts = labels.to(device), texts.to(device)
            preds = model(texts).squeeze()
            loss = criterion(preds, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

train_model()
