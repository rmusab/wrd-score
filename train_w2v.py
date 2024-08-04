import os
import torch
from gensim.models import Word2Vec
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")

# Function to recursively find all .java files
def find_java_files(directory):
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
        if len(java_files) > 50000:
            break
    return java_files

# Function to read and preprocess content of .java files
def read_and_preprocess_files(file_list, tokenizer):
    documents = []
    for file in file_list:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Use the tokenizer to preprocess the content
                tokens = tokenizer.tokenize(content)
                documents.append(tokens)
        except UnicodeDecodeError:
            # Retry with a different encoding if UTF-8 fails
            try:
                with open(file, 'r', encoding='latin-1') as f:
                    content = f.read()
                    tokens = tokenizer.tokenize(content)
                    documents.append(tokens)
            except UnicodeDecodeError:
                print(f"Skipping file due to encoding issues: {file}")
    return documents

# Main function to train Word2Vec model
def train_word2vec_model(directory, tokenizer, window_size=2, save_path='word2vec.model'):
    # Find all .java files in the directory and its subdirectories
    java_files = find_java_files(directory)
    print(f"Found {len(java_files)} .java files")

    # print(f"Taking the first 10000")
    # java_files = java_files[:10000]

    # Read and preprocess the content of the .java files
    documents = read_and_preprocess_files(java_files, tokenizer)

    # Train Word2Vec model
    model = Word2Vec(sentences=documents, vector_size=100, window=window_size, min_count=1, workers=4)
    
    # Save the model
    model.save(save_path)
    print(f"Word2Vec model saved to {save_path}")

    return model

# Path to the java-med directory
directory_path = 'Data/java-med'

# Train the model and save it
trained_model = train_word2vec_model(directory_path, tokenizer, window_size=2)

# Save the vocabulary and embeddings separately if needed
vocab_path = 'word2vec_vocab.txt'
embeddings_path = 'word2vec_embeddings.txt'

# Save vocabulary
with open(vocab_path, 'w', encoding='utf-8') as f:
    for word in trained_model.wv.index_to_key:
        f.write(f"{word}\n")

# Save embeddings
with open(embeddings_path, 'w', encoding='utf-8') as f:
    for word in trained_model.wv.index_to_key:
        embedding = trained_model.wv[word]
        f.write(f"{word} {' '.join(map(str, embedding))}\n")

print(f"Vocabulary saved to {vocab_path}")
print(f"Embeddings saved to {embeddings_path}")