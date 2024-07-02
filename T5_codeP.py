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
    code = example["func_code_string"]
    inputs = tokenizer.encode(code, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    embeddings = outputs  # The output is already the required embedding
    return embeddings.cpu().numpy().squeeze()

# Step 3: Iterate over the dataset and generate embeddings
embeddings = []
for example in dataset:
    embedding = process_example(example)
    embeddings.append(embedding)

# Step 4: Prepare matrix for SVD
embedding_matrix = np.stack(embeddings, axis=0)  # Shape: (num_examples, embedding_size)

# Step 5: Perform SVD
try:
    U, S, VT = svd(embedding_matrix, full_matrices=False)
    print(f'U shape: {U.shape}')
    print(f'S shape: {S.shape}')
    print(f'VT shape: {VT.shape}')
    
    # Step 6: Check the condition on the diagonal values
    total_sum = np.sum(S)
    cumulative_sum = 0
    iterations = 0
    for singular_value in S:
        cumulative_sum += singular_value
        iterations += 1
        if cumulative_sum / total_sum >= 0.99:
            break
    
    print(f'Number of iterations required: {iterations}')
except ValueError as e:
    print(f"Error in SVD computation: {e}")
