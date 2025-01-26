import numpy as np
import csv
import json
from sentence_transformers import SentenceTransformer, util
import torch  # To handle tensor data types

# Load a pre-trained model for sentence embeddings (local model)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose another model as well

# Function to fetch embeddings for a query using the local model
def get_embeddings(text):
    return model.encode(text)  # No need to convert to list if you're working directly with embeddings

# Function to get the best answer from CSV based on cosine similarity
def get_answer_from_csv(user_input):
    # Compute the embedding for the user's input
    user_embedding = get_embeddings(user_input)
    
    # Convert user_embedding to a PyTorch tensor with float32 dtype if it's not already
    user_embedding = torch.tensor(user_embedding, dtype=torch.float32)
    
    # Read questions, answers, and embeddings from a CSV file (adjust path if needed)
    questions_answers = []
    with open('faq_with_embeddings.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions_answers.append(row)
    
    best_similarity = -1
    best_answer = None
    
    # Loop through each Q&A pair and compare cosine similarity
    for qa in questions_answers:
        question = qa["question"]
        answer = qa["answer"]
        
        # Parse embeddings from the CSV (stored as a string, so we need to convert it to a numpy array)
        stored_embedding = np.array(json.loads(qa["embeddings"]))  # Embeddings are stored as a JSON string in the CSV
        
        # Convert stored_embedding to torch tensor and ensure the same dtype as user_embedding
        stored_embedding = torch.tensor(stored_embedding, dtype=torch.float32)
        
        # Calculate cosine similarity between user query and stored question embeddings using pytorch_cos_sim
        similarity = util.pytorch_cos_sim(user_embedding, stored_embedding)[0][0].item()  # Convert tensor to scalar
        
        # If this similarity is the best so far, store it
        if similarity > best_similarity:
            best_similarity = similarity
            best_answer = answer
    
    return best_answer

# Example usage:
user_input = "comment mettre à jour mes informations personnelles sur le système de l'établissement ?"

# Get the answer
answer = get_answer_from_csv(user_input)

# Print the answer
print(f"Answer: {answer}")
