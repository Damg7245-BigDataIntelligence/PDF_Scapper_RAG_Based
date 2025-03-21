import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Optional imports for different vector DBs
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Pinecone not available. Install with: pip install pinecone-client")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

# Load environment variables
load_dotenv()

class VectorStorageService:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector storage service with support for multiple backends
        
        Args:
            embedding_model: The model to use for embeddings
        """
        # Get API tokens from environment
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer(embedding_model, token=self.hf_token)
            self.model_name = embedding_model
            print(f"Embedding model initialized: {self.model_name}")
        except Exception as e:
            print(f"Error initializing embedding model: {str(e)}")
            # Use a simpler fallback model if available
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2", token=self.hf_token)
                self.model_name = "all-MiniLM-L6-v2"
                print(f"Using fallback model: {self.model_name}")
            except Exception as e2:
                raise Exception(f"Failed to initialize embedding model: {str(e2)}")
        
        # Set up local storage
        self.embeddings_path = Path("data/embeddings.json")
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends based on availability
        self.backends = {}
        
        # Always initialize local JSON storage
        self.backends["json"] = self._load_json_embeddings()
        
        # Initialize Pinecone if available
        if PINECONE_AVAILABLE and self.pinecone_api_key:
            try:
                # Initialize Pinecone using the latest API
                pc = Pinecone(api_key=self.pinecone_api_key)
                index_name = "nvidia-financials"
                
                # Check if index exists, create if not
                if index_name not in [idx["name"] for idx in pc.list_indexes()]:
                    pc.create_index(
                        name=index_name,
                        dimension=self.model.get_sentence_embedding_dimension(),
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                
                self.backends["pinecone"] = {
                    "index": pc.Index(index_name),
                    "info": {"index_name": index_name}
                }
                print(f"Pinecone initialized with index: {index_name}")
            except Exception as e:
                print(f"Error initializing Pinecone: {str(e)}")
        
        # Initialize ChromaDB if available
        if CHROMADB_AVAILABLE:
            try:
                chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
                os.makedirs(chroma_persist_dir, exist_ok=True)
                
                print(f"ChromaDB will store data in: {os.path.abspath(chroma_persist_dir)}")
        
                chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
                collection_name = "nvidia-financials"
                
                # Get or create collection
                self.backends["chromadb"] = {
                    "collection": chroma_client.get_or_create_collection(name=collection_name),
                    "info": {"collection_name": collection_name, "persist_dir": chroma_persist_dir}
                }
                print(f"ChromaDB initialized with persistant collection: {collection_name}")
            except Exception as e:
                print(f"Error initializing ChromaDB: {str(e)}")
    
    def _load_json_embeddings(self) -> Dict[str, Any]:
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
    
    def _save_json_embeddings(self) -> None:
        """Save embeddings to JSON file"""
        with open(self.embeddings_path, 'w') as f:
            json.dump(self.backends["json"], f)
    
    def compute_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of text chunks"""
        # Convert the embeddings to Python lists for JSON serialization
        return [embedding.tolist() for embedding in self.model.encode(chunks)]
    
    def store_document(self, document_id: str, chunks: List[str], metadata: Dict[str, Any]) -> None:
        """
        Store document chunks and their embeddings in all available backends
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of text chunks to store
            metadata: Additional metadata for the document
        """
        # Compute embeddings for all chunks
        chunk_embeddings = self.compute_embeddings(chunks)
        
        # Store in JSON
        self._store_in_json(document_id, chunks, chunk_embeddings, metadata)
        
        # Store in Pinecone if available
        if "pinecone" in self.backends:
            self._store_in_pinecone(document_id, chunks, chunk_embeddings, metadata)
        
        # Store in ChromaDB if available
        if "chromadb" in self.backends:
            self._store_in_chromadb(document_id, chunks, chunk_embeddings, metadata)
        
        print(f"Document {document_id} stored in all available backends")
    
    def _store_in_json(self, document_id: str, chunks: List[str], 
                       chunk_embeddings: List[List[float]], metadata: Dict[str, Any]) -> None:
        """Store document in JSON backend"""
        # Check if document already exists
        if document_id in self.backends["json"]["documents"]:
            print(f"Document {document_id} already exists in JSON storage. Updating...")
        
        # Create chunk data with text and embeddings
        chunk_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_data.append({
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": chunk,
                "embedding": embedding
            })
        
        # Add to embeddings data structure (overwrites if exists)
        self.backends["json"]["documents"][document_id] = {
            "metadata": metadata,
            "chunks": chunk_data,
            "added_at": datetime.now().isoformat()
        }
        
        # Update last_updated timestamp
        self.backends["json"]["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save to disk
        self._save_json_embeddings()
        print(f"Stored {len(chunks)} chunks in JSON embeddings file")
    
    def _store_in_pinecone(self, document_id: str, chunks: List[str],
                          chunk_embeddings: List[List[float]], metadata: Dict[str, Any]) -> None:
        """Store document in Pinecone backend"""
        try:
            vectors_to_upsert = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Create vector object for Pinecone
                vector = {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "document_id": document_id,
                        "chunk_index": i,
                        **metadata
                    }
                }
                
                vectors_to_upsert.append(vector)
            
            # Upsert in batches to avoid request size limits
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.backends["pinecone"]["index"].upsert(batch)
            
            print(f"Stored {len(chunks)} chunks in Pinecone")
        except Exception as e:
            print(f"Error storing in Pinecone: {str(e)}")
    
    def _store_in_chromadb(self, document_id: str, chunks: List[str],
                          chunk_embeddings: List[List[float]], metadata: Dict[str, Any]) -> None:
        """Store document in ChromaDB backend"""
        try:
            # Prepare IDs, embeddings, metadatas for ChromaDB
            ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            embeddings = chunk_embeddings
            
            # Copy metadata for each chunk
            metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["document_id"] = document_id
                chunk_metadata["chunk_index"] = i
                chunk_metadata["text"] = chunk
                metadatas.append(chunk_metadata)
            
            # Check if document already exists in ChromaDB
            try:
                # Get the first chunk ID to check if it exists
                test_id = f"{document_id}_chunk_0"
                existing_docs = self.backends["chromadb"]["collection"].get(ids=[test_id])
                
                if existing_docs and len(existing_docs['ids']) > 0:
                    # Document exists, delete all chunks for this document first
                    print(f"Document {document_id} already exists in ChromaDB. Removing old chunks...")
                    
                    # Get all chunks for this document
                    query_results = self.backends["chromadb"]["collection"].get(
                        where={"document_id": document_id}
                    )
                    
                    if query_results and len(query_results['ids']) > 0:
                        # Delete existing chunks
                        self.backends["chromadb"]["collection"].delete(
                            ids=query_results['ids']
                        )
                        print(f"Removed {len(query_results['ids'])} existing chunks from ChromaDB")
            except Exception as check_error:
                # If there's an error checking, just proceed with adding
                print(f"Error checking for existing document: {check_error}")
            
            # Now add the new chunks
            self.backends["chromadb"]["collection"].add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=chunks
            )
            
            print(f"Stored {len(chunks)} chunks in ChromaDB")
        except Exception as e:
            print(f"Error storing in ChromaDB: {str(e)}")