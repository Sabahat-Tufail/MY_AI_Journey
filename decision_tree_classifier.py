import math
from collections import Counter

# Sample dataset: [Outlook, Temperature, Play]
dataset = [
    ['Sunny', 'Hot', 'No'],
    ['Sunny', 'Cool', 'No'],
    ['Overcast', 'Hot', 'Yes'],
    ['Rain', 'Mild', 'Yes'],
    ['Rain', 'Cool', 'Yes'],
    ['Rain', 'Cool', 'No'],
    ['Overcast', 'Cool', 'Yes'],
    ['Sunny', 'Mild', 'No'],
    ['Sunny', 'Cool', 'Yes'],
    ['Rain', 'Mild', 'Yes'],
]

# Calculate entropy of a list of labels
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

# Find the best feature index to split on
def best_feature(data):
    base_entropy = entropy([row[-1] for row in data])
    best_gain = 0
    best_index = -1
    for i in range(len(data[0]) - 1):  # Exclude label
        values = set(row[i] for row in data)
        split_entropy = 0
        for val in values:
            subset = [row for row in data if row[i] == val]
            split_entropy += len(subset)/len(data) * entropy([r[-1] for r in subset])
        gain = base_entropy - split_entropy
        if gain > best_gain:
            best_gain = gain
            best_index = i
    return best_index

# Build the tree recursively
def build_tree(data):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]  # All same class

    if len(data[0]) == 1:
        return Counter(labels).most_common(1)[0][0]  # Majority class

    index = best_feature(data)
    tree = {}
    values = set(row[index] for row in data)
    for val in values:
        subset = [row[:index] + row[index+1:] for row in data if row[index] == val]
        tree[(index, val)] = build_tree(subset)
    return tree

# Predict using the built tree
def predict(tree, sample, features_left):
    while isinstance(tree, dict):
        for (idx, val), subtree in tree.items():
            if sample[idx] == val:
                new_sample = sample[:idx] + sample[idx+1:]
                new_features_left = features_left.copy()
                if idx in new_features_left:
                    new_features_left.remove(idx)
                return predict(subtree, new_sample, new_features_left)
        return None  # Value not found
    return tree


# Train
tree = build_tree(dataset)
print("Decision Tree:", tree)

# Test prediction
sample = ['Sunny', 'Cool']  # Should match something from data
print("Prediction:", predict(tree, sample, [0, 1]))
