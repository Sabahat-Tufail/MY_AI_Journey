

import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')


# Sample text
text = "Hello! I love AI and ML. I am learning Python programming. I am also learning Natural Language Processing (NLP). Visit https://example.com for more info."

# Step 1: Raw Text Noise Removal
# Remove URLs
text = re.sub(r'http\S+', '', text)
# Remove numbers
text = re.sub(r'\d+', '', text)
# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))
# Convert to lowercase
text = text.lower()

# Step 2: Tokenization
# Sentence Tokenization
sentences = sent_tokenize(text)
# Word Tokenization
words = word_tokenize(text)

# Step 3: Remove Stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Step 4: Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Step 5: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

# Step 6: POS Tagging
pos_tags = pos_tag(lemmatized_words)


# Output all the results
print("Cleaned Text: ", text)
print("\nTokenized Sentences: ", sentences)
print("\nTokenized Words: ", words)
print("\nFiltered Words (Stopwords Removed): ", filtered_words)
print("\nStemmed Words: ", stemmed_words)
print("\nLemmatized Words: ", lemmatized_words)
print("\nPOS Tags: ", pos_tags)

