from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables to get the Hugging Face token
load_dotenv()
def generate_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings using the specified model"""
    # Load model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    embedding = model.encode(text).tolist()
    
    return embedding