import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model for embedding (ensure it's the same model used to encode the FAQ data)
def top_five(input):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your model

    # Load the FAQ CSV file with embeddings
    faq_df = pd.read_csv('faq.csv')
    print(faq_df.columns)
    # Convert the embeddings from string back to numpy arrays
    faq_df['Embeddings'] = faq_df['Embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

    # Define a function to calculate cosine similarity
    def cosine_similarity_manual(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    # Encode the user input to get the embedding
    user_embedding = model.encode([input], convert_to_tensor=True).numpy().flatten()

    # Compute cosine similarity between the user input embedding and all FAQ embeddings
    cos_similarities = []
    for idx, row in faq_df.iterrows():
        faq_embedding = np.array(row['Embeddings'])  # Convert the embedding back to numpy array
        sim = cosine_similarity_manual(user_embedding, faq_embedding)  # Use manual cosine similarity
        cos_similarities.append((row['Context'], row['Response'],sim))

    # Sort the similarities in descending order to get the most similar questions
    cos_similarities.sort(key=lambda x: x[2], reverse=True)

    # Get the top 3 similar FAQs
    top_three = cos_similarities[:2]
    
    # Prepare the output by extracting questions and answers
    top_two_questions = [entry[0] for entry in top_three]
    top_two_answers = [entry[1] for entry in top_three]
    # Optionally, print the results
    print(top_two_questions)
    print(top_two_answers)
    
    return [top_two_questions, top_two_answers]
top_five("hello")
