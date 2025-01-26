import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model for embedding (ensure it's the same model used to encode the FAQ data)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your model

# Load the FAQ CSV file with embeddings
faq_df = pd.read_csv('faq.csv')

# Convert the embeddings from string back to numpy arrays
faq_df['embeddings'] = faq_df['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Define a function to calculate cosine similarity
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Get the user input
user_input = "Quelle est la mission de ihec ?"

# Encode the user input to get the embedding
user_embedding = model.encode([user_input], convert_to_tensor=True).numpy().flatten()

# Compute cosine similarity between the user input embedding and all FAQ embeddings
cos_similarities = []
for idx, row in faq_df.iterrows():
    faq_embedding = np.array(row['embeddings'])  # Convert the embedding back to numpy array
    sim = cosine_similarity_manual(user_embedding, faq_embedding)  # Use manual cosine similarity
    cos_similarities.append((row['id'], row['category'], row['question'], row['answer'], sim))

# Sort the similarities in descending order to get the most similar question
cos_similarities.sort(key=lambda x: x[4], reverse=True)

# Get the most similar FAQ entry
max_sim = cos_similarities[0]

# Output the result
print(f"Most similar question: {max_sim[2]}")
print(f"Answer: {max_sim[3]}")
print(f"Cosine similarity: {max_sim[4]}")
