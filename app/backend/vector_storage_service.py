import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Global model cache for efficiency
_model = None

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Get or initialize the embedding model"""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(model_name)
            print(f"Embedding model initialized: {model_name}")
        except Exception as e:
            print(f"Error initializing embedding model: {str(e)}")
            try:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Using fallback model: all-MiniLM-L6-v2")
            except Exception as e2:
                raise Exception(f"Failed to initialize embedding model: {str(e2)}")
    return _model

def generate_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings for text using the specified model"""
    model = get_embedding_model(model_name)
    # Handle single string or list of strings
    if isinstance(text, str):
        return model.encode(text).tolist()
    else:
        return model.encode(text).tolist()

def store_in_chromadb(embeddings_data, collection_name="nvidia_financials"):
    """Store embeddings data in ChromaDB"""
    try:
        # First try to get the app's client if we're in a FastAPI context
        import inspect
        from fastapi import FastAPI
        
        app = None
        for frame_info in inspect.stack():
            if 'app' in frame_info.frame.f_locals:
                potential_app = frame_info.frame.f_locals['app']
                if isinstance(potential_app, FastAPI):
                    app = potential_app
                    break
        
        if app and hasattr(app, 'chroma_client'):
            # Use the app's persistent client
            client = app.chroma_client
            print("Using app's persistent ChromaDB client for storage")
        else:
            # Fall back to in-memory client like in chromadbtest.py
            client = chromadb.Client()
            print("Using in-memory ChromaDB client for storage")
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
        except:
            collection = client.create_collection(name=collection_name)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for i, item in enumerate(embeddings_data):
            chunk_id = f"{item['metadata']['document_id']}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(item['content'])
            embeddings.append(item['embedding'])
            metadatas.append(item['metadata'])
        
        # Add to collection - using the same format as chromadbtest.py
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"Stored {len(embeddings_data)} chunks in ChromaDB collection '{collection_name}'")
        return True
    except Exception as e:
        print(f"Error storing in ChromaDB: {str(e)}")
        return False

def search_chromadb(query, collection_name="nvidia_financials", where_filter=None, top_k=5):
    """Search for similar documents in ChromaDB"""
    try:
        # First try to get the app's client if we're in a FastAPI context
        import inspect
        from fastapi import FastAPI
        
        app = None
        for frame_info in inspect.stack():
            if 'app' in frame_info.frame.f_locals:
                potential_app = frame_info.frame.f_locals['app']
                if isinstance(potential_app, FastAPI):
                    app = potential_app
                    break
        
        if app and hasattr(app, 'chroma_client'):
            # Use the app's persistent client
            client = app.chroma_client
            print("Using app's persistent ChromaDB client for search")
        else:
            # Fall back to in-memory client like in chromadbtest.py
            client = chromadb.Client()
            print("Using in-memory ChromaDB client for search")
        
        # Get the collection
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            print(f"Collection '{collection_name}' not found: {str(e)}")
            return []
        
        # Generate embedding for query
        query_embedding = generate_embeddings(query)
        
        # Execute query - use the same format as in chromadbtest.py
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        
        if results and 'documents' in results and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i] if 'metadatas' in results and len(results['metadatas']) > 0 else {}
                distance = results['distances'][0][i] if 'distances' in results and len(results['distances']) > 0 else 1.0
                
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "document_id": metadata.get("document_id", "unknown"),
                    "quarter": metadata.get("quarter", "unknown"),
                    "similarity": 1.0 - float(distance)  # Convert distance to similarity
                })
        
        return formatted_results
    except Exception as e:
        print(f"Error searching ChromaDB: {str(e)}")
        return []

def store_in_pinecone(embeddings_data, index_name="nvidia-financials"):
    """Store embeddings data in Pinecone"""
    try:
        # Initialize Pinecone if not already initialized
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        
        if not api_key:
            raise ValueError("Pinecone API key not configured")
            
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        if index_name not in [idx["name"] for idx in pc.list_indexes()]:
            # Get dimension from first embedding
            dimension = len(embeddings_data[0]['embedding'])
            
            # Create index with the right dimension
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Get index
        index = pc.Index(index_name)
        
        # Prepare vectors for Pinecone
        vectors = []
        
        for i, item in enumerate(embeddings_data):
            chunk_id = f"{item['metadata']['document_id']}_chunk_{i}"
            vector = {
                    "id": chunk_id,
                "values": item['embedding'],
                    "metadata": {
                    "text": item['content'],
                    **item['metadata']
                }
            }
            vectors.append(vector)
        
        # Upsert in batches
            batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"Stored {len(embeddings_data)} chunks in Pinecone index '{index_name}'")
        return True
    except Exception as e:
        print(f"Error storing in Pinecone: {str(e)}")
        return False

def search_pinecone(query, index_name="nvidia-financials", filter_dict=None, top_k=5):
    """Search for similar documents in Pinecone"""
    try:
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            raise ValueError("Pinecone API key not configured")
            
        pc = Pinecone(api_key=api_key)
        
        # Generate embedding for query
        query_embedding = generate_embeddings(query)
        print("Query embedding: ", query_embedding)
        # Get Pinecone index
        index = pc.Index(index_name)
        
        # Query the index
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        print("Results from Pinecone: ", results)
        # Format results
        formatted_results = []
        
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            
            formatted_results.append({
                "text": metadata.get("text", ""),
                "document_id": metadata.get("document_id", "unknown"),
                "quarter": metadata.get("quarter", "unknown"),
                "similarity": float(match.get("score", 0.0))
            })
        
        return formatted_results
    except Exception as e:
        print(f"Error searching Pinecone: {str(e)}")
        return []

def store_embeddings_json(embeddings_data, file_path="data/embeddings/nvidia_embeddings.json"):
    """Store embeddings in a JSON file"""
    try:
        # Create directory if it doesn't exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if file exists
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            # Initialize new data structure
            data = []
        
        # Add new embeddings
        for item in embeddings_data:
            data.append({
                "content": item['content'],
                "embedding": item['embedding'],
                "metadata": item['metadata']
            })
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Stored {len(embeddings_data)} chunks in {file_path}")
        return True
    except Exception as e:
        print(f"Error storing in JSON: {str(e)}")
        return False

def search_embeddings_json(query, embeddings_file="data/embeddings/nvidia_embeddings.json", quarter_filter=None, top_k=5):
    """Search embeddings.json file for relevant chunks"""
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file not found: {embeddings_file}")
        return []
    
    try:
        # Load embeddings
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        
        # Filter by quarters if provided
        if quarter_filter and len(quarter_filter) > 0:
            filtered_data = [
                item for item in data 
                if "metadata" in item and 
                "quarter" in item["metadata"] and 
                item["metadata"]["quarter"] in quarter_filter
            ]
        else:
            filtered_data = data
        
        # If no data after filtering, return empty results
        if not filtered_data:
            return []
        
        # Generate embedding for the question
        query_embedding = generate_embeddings(query)
        
        # Compute cosine similarity with all chunks
        def cosine_similarity(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        similarities = []
        for item in filtered_data:
            if "embedding" in item:
                similarity = cosine_similarity(
                    np.array(query_embedding), 
                    np.array(item["embedding"])
                )
                
                similarities.append({
                    "text": item["content"],
                    "document_id": item.get("metadata", {}).get("document_id", "unknown"),
                    "quarter": item.get("metadata", {}).get("quarter", "unknown"),
                    "filename": item.get("metadata", {}).get("filename", "unknown"),
                    "similarity": float(similarity)
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k results
        return similarities[:top_k]
        
    except Exception as e:
        print(f"Error searching embeddings: {str(e)}")
        return []