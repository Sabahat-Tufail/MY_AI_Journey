
'''import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string




text ="Hello ! I love AI and ML. I am learning Python programming. I am also learning Natural Language Processing (NLP)."

tokens= word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words ]

# Remove punctuation
filtered_tokens = [word for word in filtered_tokens if word not in string.punctuation]
print(filtered_tokens)

# code 2

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')


lemmatizer= WordNetLemmatizer()
text = "The cats are playing with the mice."

tokens= word_tokenize(text.lower())

lemmatize_token= [lemmatizer.lemmatize(word) for word in tokens]

print(lemmatize_token)
#TASK 1
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string




text ="A quick # brown 'fox'. Jumps over.@The 'lazy' DOG !?"


tokens= word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words ]

# Remove punctuation
filtered_tokens = [word for word in filtered_tokens if word not in string.punctuation]
print(filtered_tokens)

#TASK 2
# Step 1: Import spaCy and load the model
import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Step 2: Define the text
text = "The cats are playing in the garden. They were running after the mouse."

# Process the text with spaCy to tokenize and lemmatize
doc = nlp(text)

# Step 3: Output the lemmatized words
lemmatized_words = [token.lemma_ for token in doc]

print("Lemmatized words:", lemmatized_words)'''

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk import pos_tag

# Sample text
text = "Hello! I love AI and ML. I am learning Python programming. I am also learning Natural Language Processing (NLP)."

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word Tokenization
words = word_tokenize(text)
print("Words:", words)

# Stemming
stemmer = PorterStemmer()
words_to_stem = ["running", "runs", "runner", "happiness"]
print("Stemming Results:")
for word in words_to_stem:
    print(f"{word} -> {stemmer.stem(word)}")

# Stopwords and punctuation removal
text1 = "Hello! Hows the weather"
text1 = text1.lower()
text1 = text1.translate(str.maketrans('', '', string.punctuation))
words = word_tokenize(text1)
words = [word for word in words if word not in stopwords.words('english')]

print("After Removing Stopwords:", words)

# Regular Expression for Email and Phone Number
text2 = "Contact me at sabahatxwat2gmail.com or call 123-4556-666"

# Corrected email regex
emails = re.findall(r'\S+@\S+', text2)
print("Emails:", emails)

# Corrected phone number regex
phone_number = re.findall(r'\d{3}-\d{4}-\d{3}', text2)
print("Phone Numbers:", phone_number)

# Part-of-Speech Tagging
#nltk.download('averaged_perceptron_tagger')  # Uncomment if not already downloaded
text3 = "I am going home"
words2 = word_tokenize(text3)
import nltk
nltk.download('averaged_perceptron_tagger')


pos_tags = pos_tag(words2)
print("POS Tags:", pos_tags)


import re

# Camel case text
text4 = "ThisIsValidTask"

# Function to split camel case text into words
def split_camel_case(text):
    # Split at each uppercase letter and keep the word
    return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', text)

# Use the function to split the text
split_words = split_camel_case(text4)

# Print the result
print("Split Words:", split_words)
