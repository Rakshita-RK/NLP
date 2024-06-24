from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from scipy.linalg import svd

# Step 1: Load the CodeSearchNet dataset
dataset = load_dataset("code_search_net", "python", split="test")

# Step 2: Initialize the T5 model and tokenizer
checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# Function to process a single example and get embeddings
def process_example(example):
    code = example["code"]
    inputs = tokenizer.encode(code, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    # Extract the last hidden state (embeddings)
    embeddings = outputs.last_hidden_state  # Shape: (1, sequence_length, embedding_size)
    # Mean pooling to get a single embedding per function
    mean_embedding = embeddings.mean(dim=1)  # Shape: (1, embedding_size)
    return mean_embedding.cpu().numpy().squeeze()

# Step 3: Iterate over the dataset and generate embeddings
embeddings = []
for example in dataset:
    embedding = process_example(example)
    embeddings.append(embedding)

# Step 4: Prepare matrix for SVD
embedding_matrix = np.stack(embeddings, axis=0)  # Shape: (num_examples, embedding_size)

# Step 5: Perform SVD
U, S, VT = svd(embedding_matrix, full_matrices=False)

# Print shapes of SVD components
print(f'U shape: {U.shape}')
print(f'S shape: {S.shape}')
print(f'VT shape: {VT.shape}')

