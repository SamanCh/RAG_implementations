#tst_rag

import os
import numpy as np
from typing import List, Dict, Any, Tuple
import re

# For simplicity, we're simulating embeddings without external dependencies
# In a real implementation, you would use a model like OpenAI's embeddings or SentenceTransformers

class SimpleEmbedder:
    """A simple embedder that creates vector representations of text"""
    
    def __init__(self, dimension: int = 384):
        """Initialize with vector dimension"""
        self.dimension = dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create a deterministic but simplistic embedding based on text features"""
        # This is a very naive approach - just for demonstration purposes
        # Real systems would use proper embedding models
        
        # Create a seed from the text to make embeddings deterministic
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        
        # Generate a random vector
        embedding = np.random.normal(0, 1, self.dimension)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

class DocumentChunker:
    """Split documents into manageable chunks"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize with chunk size and overlap parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        # Simple character-based chunking
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at a period or space to keep context intact
            if end < len(text):
                # Look for the last period or space within the current chunk
                last_period = text.rfind('.', start, end)
                last_space = text.rfind(' ', start, end)
                
                if last_period > start + self.chunk_size // 2:
                    end = last_period + 1  # Include the period
                elif last_space > start + self.chunk_size // 2:
                    end = last_space + 1  # Include the space
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a document and preserve metadata"""
        chunked_documents = []
        chunks = self.chunk_text(document['content'])
        
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                'id': f"{document['id']}_chunk_{i}",
                'content': chunk,
                'metadata': {
                    'source': document['metadata']['source'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_id': document['id']
                }
            })
        
        return chunked_documents

class VectorStore:
    """Simple in-memory vector store"""
    
    def __init__(self, embedder: SimpleEmbedder):
        """Initialize with an embedder"""
        self.embedder = embedder
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store"""
        for doc in documents:
            embedding = self.embedder.embed_text(doc['content'])
            self.documents.append(doc)
            self.embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        if not self.documents:
            return []
        
        query_embedding = self.embedder.embed_text(query)
        
        # Calculate cosine similarity
        similarities = [
            np.dot(query_embedding, doc_embedding)
            for doc_embedding in self.embeddings
        ]
        
        # Get top k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'document': self.documents[idx],
                'score': similarities[idx]
            }
            for idx in top_indices
        ]

class SimpleGenerator:
    """Generate responses based on context and query"""
    
    def __init__(self):
        """Initialize the generator"""
        pass
    
    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate a response based on the query and retrieved documents"""
        # In a real implementation, this would use an LLM like GPT
        # Here we're just demonstrating the structure
        
        if not context_docs:
            return "I don't have enough information to answer that question."
        
        # Extract text from documents
        context_texts = [doc['document']['content'] for doc in context_docs]
        
        # Combine context into one string
        full_context = "\n\n".join(context_texts)
        
        # Create a simple response template
        response = f"Based on the retrieved information, I can answer your query: '{query}'\n\n"
        
        # In a real system, we would pass this to an LLM
        # For demo purposes, we'll just return a simple response with context snippets
        response += "Here's what I found in the documents:\n\n"
        
        for i, doc in enumerate(context_docs):
            snippet = doc['document']['content']
            source = doc['document']['metadata']['source']
            score = doc['score']
            
            # Truncate long snippets
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
                
            response += f"[Document {i+1} from {source} (relevance: {score:.2f})]\n{snippet}\n\n"
        
        return response.strip()

class RagSystem:
    """Complete RAG system combining all components"""
    
    def __init__(self, 
                 embedder: SimpleEmbedder = None,
                 chunker: DocumentChunker = None,
                 vector_store: VectorStore = None,
                 generator: SimpleGenerator = None):
        """Initialize the RAG system with optional components"""
        self.embedder = embedder or SimpleEmbedder()
        self.chunker = chunker or DocumentChunker()
        self.vector_store = vector_store or VectorStore(self.embedder)
        self.generator = generator or SimpleGenerator()
    
    def add_document(self, content: str, source: str = "unknown") -> None:
        """Add a document to the system"""
        doc_id = f"doc_{len(self.vector_store.documents)}"
        
        document = {
            'id': doc_id,
            'content': content,
            'metadata': {
                'source': source
            }
        }
        
        chunked_docs = self.chunker.chunk_document(document)
        self.vector_store.add_documents(chunked_docs)
        
        return f"Added document with ID {doc_id} from source {source}"
    
    def query(self, query: str, top_k: int = 3) -> str:
        """Process a query through the RAG pipeline"""
        # Search for relevant documents
        relevant_docs = self.vector_store.search(query, top_k=top_k)
        
        # Generate response
        response = self.generator.generate(query, relevant_docs)
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = RagSystem()
    
    # Add some sample documents
    rag.add_document(
        "RAG (Retrieval-Augmented Generation) is a technique that enhances large language models by retrieving relevant information from external knowledge sources. This approach combines the strengths of retrieval-based and generation-based methods, allowing models to access up-to-date information beyond their training data and generate more accurate, factual responses.",
        source="RAG Introduction"
    )
    
    rag.add_document(
        "The RAG architecture typically consists of several components: a retriever that finds relevant documents from a knowledge base, and a generator that creates responses based on both the query and the retrieved documents. This approach helps reduce hallucinations in LLMs by grounding responses in factual information.",
        source="RAG Architecture"
    )
    
    rag.add_document(
        "Vector databases are essential for efficient RAG systems. They store document embeddings (numerical representations of text) and enable semantic search through similarity metrics like cosine similarity. Popular vector databases include Pinecone, Weaviate, and FAISS.",
        source="Vector Databases"
    )
    
    # Query the system
    query = "What is RAG and how does it work?"
    response = rag.query(query)
    
    print(f"Query: {query}\n")

    print(f"Response:\n{response}")
