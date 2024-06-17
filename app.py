import streamlit as st
import numpy as np
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from scipy.spatial.distance import cosine as cosine_similarity

# Specific model being used
MODEL_NAME = 'textembedding-gecko@003' 

# Function to get embeddings for words
def embed_text(texts: list, model_name: str = MODEL_NAME) -> list:
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        inputs = [TextEmbeddingInput(text) for text in texts]
        embeddings = model.get_embeddings(inputs)
        return [np.array(embedding.values) for embedding in embeddings]
    except Exception as e:
        raise RuntimeError(f"Error embedding texts: {e}")

# Function to calculate cosine similarity between two vectors
def cosine_similarity_percentage(vec1: np.ndarray, vec2: np.ndarray) -> float:
    similarity = 1 - cosine_similarity(vec1, vec2)
    return similarity * 100

# Streamlit application
st.title('Word Similarity Calculator!')

# Input fields for the two words
word1 = st.text_input('Enter the first word:')
word2 = st.text_input('Enter the second word:')

# Calculate similarity button
if st.button('Calculate Similarity'):
    if word1 and word2:
        try:
            # Embed the words
            embeddings = embed_text([word1, word2], MODEL_NAME)

            # Calculate cosine similarity percentage
            similarity_percentage = cosine_similarity_percentage(embeddings[0], embeddings[1])

            # Display similarity percentage
            st.write(f"The similarity between '{word1}' and '{word2}' is: {similarity_percentage:.2f}%")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning('Please enter both words to calculate similarity!')