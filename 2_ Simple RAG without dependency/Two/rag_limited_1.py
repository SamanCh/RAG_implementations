import os
import numpy as np
from typing import List, Dict, Any, Tuple
import re

# For simplicity, we're simulating embeddings without external dependencies
# In a real implementation, you would use a model like OpenAI's embeddings or SentenceTransformers

class SimpleEmbedder:
    """A simple embedder that creates vector representations of text"""
    
    def __init__(self, dimension: int = 64):  # Reduced dimension for memory efficiency
        """Initialize with vector dimension"""
        self.dimension = dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create a deterministic but simplistic embedding based on text features"""
        # This is a very naive approach - just for demonstration purposes
        # Real systems would use proper embedding models
        
        # Truncate very long texts to avoid memory issues
        text = text[:10000] if len(text) > 10000 else text
        
        # Create a seed from the text to make embeddings deterministic
        # Use a more memory-efficient way to create a seed
        seed = sum(ord(c) for i, c in enumerate(text) if i % 5 == 0)  # Sample every 5th character
        np.random.seed(seed)
        
        # Generate a random vector
        embedding = np.random.normal(0, 1, self.dimension)
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

class DocumentChunker:
    """Split documents into manageable chunks"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize with chunk size and overlap parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks - memory efficient version"""
        # Early exit for empty text
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        # Handle very large documents by processing in smaller windows
        while start < text_length:
            # Calculate end position for this chunk
            end = min(start + self.chunk_size, text_length)
            
            # Try to end at a natural boundary if possible
            if end < text_length:
                # Create a small search window to find natural boundaries
                search_end = end
                search_start = max(start + self.chunk_size // 2, 0)
                
                # Get the substring we're searching in without loading the entire text
                search_text = text[search_start:search_end]
                
                # Find the last period or space in the search window
                last_period = search_text.rfind('.')
                last_space = search_text.rfind(' ')
                
                # Apply the offset to get the actual position in the full text
                if last_period > 0:  # If we found a period
                    end = search_start + last_period + 1  # Include the period
                elif last_space > 0:  # If we found a space
                    end = search_start + last_space + 1  # Include the space
            
            # Extract just the current chunk
            current_chunk = text[start:end]
            chunks.append(current_chunk)
            
            # Move the start position, considering overlap
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
    """Simple in-memory vector store with memory optimization"""
    
    def __init__(self, embedder: SimpleEmbedder):
        """Initialize with an embedder"""
        self.embedder = embedder
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store"""
        # Process documents in smaller batches to save memory
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            for doc in batch:
                # Only store a reference to the content to reduce memory usage
                content = doc['content']
                doc_copy = doc.copy()
                
                # For very large content, store a summary instead
                if len(content) > 1000:
                    doc_copy['content_summary'] = content[:997] + "..."
                    doc_copy['content'] = content  # Keep full content
                
                embedding = self.embedder.embed_text(content)
                self.documents.append(doc_copy)
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
        """Initialize the RAG system with optimal default parameters"""
        self.embedder = embedder or SimpleEmbedder(dimension=64)  # Smaller embedding size
        self.chunker = chunker or DocumentChunker(chunk_size=300, chunk_overlap=30)  # Smaller chunks
        self.vector_store = vector_store or VectorStore(self.embedder)
        self.generator = generator or SimpleGenerator()
        self.document_count = 0
    
    def add_document(self, content: str, source: str = "unknown") -> str:
        """Add a document to the system with better memory management"""
        # Check if content is extremely large and might cause issues
        if len(content) > 1000000:  # If over ~1MB
            print(f"Warning: Document is very large ({len(content)} chars). Processing in smaller parts.")
            # Process extremely large documents in parts
            parts = []
            chunk_size = 500000  # ~500KB chunks
            for i in range(0, len(content), chunk_size):
                parts.append(content[i:i+chunk_size])
            
            results = []
            for i, part in enumerate(parts):
                doc_id = f"doc_{self.document_count}_{i}"
                document = {
                    'id': doc_id,
                    'content': part,
                    'metadata': {
                        'source': f"{source} (part {i+1}/{len(parts)})"
                    }
                }
                
                # Process one chunk at a time to save memory
                try:
                    chunked_docs = self.chunker.chunk_document(document)
                    self.vector_store.add_documents(chunked_docs)
                    results.append(f"Added document part {i+1}/{len(parts)} with ID {doc_id}")
                except MemoryError as e:
                    return f"Memory error processing document: {str(e)}. Try with smaller documents or increase available memory."
            
            self.document_count += 1
            return "\n".join(results)
        else:
            # Normal processing for reasonably sized documents
            doc_id = f"doc_{self.document_count}"
            self.document_count += 1
            
            document = {
                'id': doc_id,
                'content': content,
                'metadata': {
                    'source': source
                }
            }
            
            try:
                chunked_docs = self.chunker.chunk_document(document)
                self.vector_store.add_documents(chunked_docs)
                return f"Added document with ID {doc_id} from source {source}"
            except MemoryError as e:
                return f"Memory error processing document: {str(e)}. Try with smaller documents or increase available memory."
    
    def query(self, query: str, top_k: int = 3) -> str:
        """Process a query through the RAG pipeline"""
        # Search for relevant documents
        relevant_docs = self.vector_store.search(query, top_k=top_k)
        
        # Generate response
        response = self.generator.generate(query, relevant_docs)
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system with memory-efficient parameters
    print("Initializing RAG system...")
    rag = RagSystem(
        embedder=SimpleEmbedder(dimension=32),  # Even smaller for testing
        chunker=DocumentChunker(chunk_size=200, chunk_overlap=20)  # Smaller chunks for testing
    )
    
    # Add some sample documents - with smaller content to avoid memory issues
    print("Adding sample documents...")
    
    try:
        rag.add_document(
            "RAG (Retrieval-Augmented Generation) is a technique that enhances large language models by retrieving relevant information from external knowledge sources.",
            source="RAG Introduction"
        )
        
        rag.add_document(
            "The RAG architecture combines a retriever and a generator. The retriever finds relevant documents, and the generator creates responses based on the query and retrieval results.",
            source="RAG Architecture"
        )
        
        rag.add_document(
            "Vector databases store document embeddings and enable semantic search. Examples include Pinecone, Weaviate, and FAISS.",
            source="Vector Databases"
        )
        
        # Query the system
        query = "What is RAG and how does it work?"
        print(f"\nQuerying: '{query}'")
        response = rag.query(query)
        
        print(f"\nResponse:\n{response}")
        
    except MemoryError as e:
        print(f"Memory error: {str(e)}")
        print("Try reducing document size or increasing available memory.")
    except Exception as e:
        print(f"Error: {str(e)}")