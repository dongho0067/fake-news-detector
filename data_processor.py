import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Detector_AI import fake_data, real_data
import string
import torch

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    return tokens

# Concatenate fake and real data
all_data = fake_data + real_data

# Create vocabulary
vocab = set()
for data in all_data:
    tokens = preprocess_text(data.text)
    vocab.update(tokens)

# Create word to index mapping
word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # Add 1 for padding index

# Convert text to tensors using word indices
def text_to_tensor(text, max_len):
    tokens = preprocess_text(text)
    token_indices = [word_to_idx.get(token, 0) for token in tokens]  # Use 0 for unknown words
    token_indices = token_indices[:max_len] + [0] * (max_len - len(token_indices))  # Padding
    return torch.tensor(token_indices, dtype=torch.long)