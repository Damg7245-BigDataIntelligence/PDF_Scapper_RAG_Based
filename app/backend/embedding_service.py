import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables to get the Hugging Face token
load_dotenv()

class EmbeddingService:
    def __init__(self):
        """Initialize the embedding service with a reliable model"""
        # Get the Hugging Face token from environment variables
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.hf_token:
            print("Warning: HUGGINGFACE_TOKEN environment variable not found. Model loading might fail.")
        
        try:
            # Use a common model with explicit token
            self.model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2', 
                token=self.hf_token
            )
            self.model_name = 'all-MiniLM-L6-v2'
            
            # Path to store the embeddings JSON file
            self.embeddings_path = Path("data/embeddings.json")
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing embeddings if available
            self.embeddings_data = self._load_embeddings()
            print(f"Embedding service initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing primary embedding model: {str(e)}")
            # Try a different model format without the organization prefix
            try:
                self.model = SentenceTransformer(
                    'all-MiniLM-L6-v2', 
                    token=self.hf_token
                )
                self.model_name = 'all-MiniLM-L6-v2'
                print(f"Using fallback model: {self.model_name}")
                
                # Path to store the embeddings JSON file
                self.embeddings_path = Path("data/embeddings.json")
                self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing embeddings if available
                self.embeddings_data = self._load_embeddings()
            except Exception as e2:
                print(f"Error initializing fallback model: {str(e2)}")
                # Try local BERT as last resort (should be available through base installation)
                self.model = SentenceTransformer('bert-base-nli-mean-tokens', token=self.hf_token)
                self.model_name = 'bert-base-nli-mean-tokens'
                print(f"Using last resort model: {self.model_name}")
                
                # Path to store the embeddings JSON file
                self.embeddings_path = Path("data/embeddings.json")
                self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing embeddings if available
                self.embeddings_data = self._load_embeddings()
    
    def _load_embeddings(self) -> Dict[str, Any]:
        """Load embeddings from JSON file if it exists"""
        if self.embeddings_path.exists():
            with open(self.embeddings_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize with empty structure
            return {
                "metadata": {
                    "model_name": self.model_name,
                    "last_updated": datetime.now().isoformat(),
                },
                "documents": {}
            }
    
    def _save_embeddings(self) -> None:
        """Save embeddings to JSON file"""
        with open(self.embeddings_path, 'w') as f:
            json.dump(self.embeddings_data, f)
    
    def create_semantic_chunks(self, text: str, min_chunk_size: int = 200, max_chunk_size: int = 500) -> List[str]:
        """
        Create semantic chunks from text based on content rather than fixed size.
        This is a simple approach that uses paragraph and sentence boundaries.
        
        Args:
            text: The text to chunk
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size and we already have content,
            # save the current chunk and start a new one
            if len(current_chunk) + len(paragraph) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Add a space between paragraphs if current chunk is not empty
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def compute_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of text chunks"""
        # Convert the embeddings to Python lists for JSON serialization
        return [embedding.tolist() for embedding in self.model.encode(chunks)]
    
    def add_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Process a document and add its embeddings to the store
        
        Args:
            document_id: Unique identifier for the document
            content: Text content of the document
            metadata: Additional metadata about the document
        """
        # Create semantic chunks
        chunks = self.create_semantic_chunks(content)
        
        # Compute embeddings for chunks
        chunk_embeddings = self.compute_embeddings(chunks)
        
        # Create chunk data with text and embeddings
        chunk_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_data.append({
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": chunk,
                "embedding": embedding
            })
        
        # Add to embeddings data structure
        self.embeddings_data["documents"][document_id] = {
            "metadata": metadata,
            "chunks": chunk_data,
            "added_at": datetime.now().isoformat()
        }
        
        # Update last_updated timestamp
        self.embeddings_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save to disk
        self._save_embeddings()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with their document info and similarity scores
        """
        # Compute query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Get all chunks from all documents
        all_chunks = []
        for doc_id, doc_data in self.embeddings_data["documents"].items():
            for chunk in doc_data["chunks"]:
                all_chunks.append({
                    "document_id": doc_id,
                    "document_metadata": doc_data["metadata"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "embedding": chunk["embedding"]
                })
        
        # Calculate cosine similarity for each chunk
        results = []
        for chunk in all_chunks:
            similarity = self._calculate_cosine_similarity(query_embedding, chunk["embedding"])
            results.append({
                "document_id": chunk["document_id"],
                "document_metadata": chunk["document_metadata"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "similarity": similarity
            })
        
        # Sort by similarity (highest first) and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Convert to numpy arrays for easier calculation
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1_np, vec2_np)
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2) 