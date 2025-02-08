from sentence_transformers import SentenceTransformer
import pandas as pd

# Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the CSV file
input_file = 'faq1.csv'  # Ensure this file exists in the same directory
df = pd.read_csv(input_file)

# Extract questions (contexts) for embedding
contexts = df['Context'].tolist()

# Calculate embeddings
embeddings = model.encode(contexts, convert_to_numpy=True)

# Add embeddings as a new column
df['Embeddings'] = embeddings.tolist()

# Save to a new CSV file
output_file = 'faq.csv'
df.to_csv(output_file, index=False)

print(f"Embeddings saved to {output_file}")
