import streamlit as st
import numpy as np
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from scipy.spatial.distance import cosine as cosine_similarity
import plotly.graph_objs as go

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
st.title('Semantic Vector Analyzer')

# Input fields for the words
word1 = st.text_input('Enter the first word:')
word2 = st.text_input('Enter the second word:')

# Calculate embeddings button
if st.button('Calculate Similarity'):
    if word1 and word2:
        try:
            # Embed the words
            embeddings = embed_text([word1, word2], MODEL_NAME)

            # Calculate cosine similarity
            similarity_percentage = cosine_similarity_percentage(embeddings[0], embeddings[1])

            # Display cosine similarity
            st.subheader(f"The similarity between '{word1}' and '{word2}' is: {similarity_percentage:.2f}%")

            # Prepare data for 3D plot
            trace1 = go.Scatter3d(
                x=[embeddings[0][0], embeddings[1][0]],
                y=[embeddings[0][1], embeddings[1][1]],
                z=[embeddings[0][2], embeddings[1][2]],
                mode='markers',
                marker=dict(
                    size=12,
                    color=['blue', 'purple'],
                    opacity=0.8
                ),
                text=[word1, word2]
            )

            data = [trace1]

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0)
            )

            fig = go.Figure(data=data, layout=layout)

            # Display 3D plot
            st.plotly_chart(fig)

        except RuntimeError as e:
            st.error(f"RuntimeError: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning('Please enter both words to calculate 3D vectors!')