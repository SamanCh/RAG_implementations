import numpy as np
import random
import re
import gc  # For explicit garbage collection

"""
Ultra-lightweight RAG implementation
- Designed for minimal memory footprint
- Efficient processing of large documents
- Simple streaming architecture
"""

class MinimalEmbedder:
    """Extremely lightweight embedder with minimal memory usage"""
    
    def __init__(self, dimension=16):  # Very small dimension for minimal memory usage
        self.dimension = dimension
        # Use a small seed dictionary for common words to make embeddings somewhat meaningful
        self.word_seeds = {
            "the": 1, "and": 2, "of": 3, "to": 4, "in": 5,
            "a": 6, "is": 7, "that": 8, "for": 9, "it": 10,
            "with": 11, "as": 12, "was": 13, "on": 14, "be": 15
        }
    
    def embed_text(self, text):
        """Create a simple deterministic embedding with minimal computation"""
        # Truncate to reasonable length
        text = text[:1000] if len(text) > 1000 else text
        
        # Extract a few key words for determinism without processing the whole text
        words = re.findall(r'\b\w+\b', text.lower())
        sample_words = words[:20] if len(words) > 20 else words
        
        # Create a simple seed based on common word presence
        seed_val = 0
        for word in sample_words:
            if word in self.word_seeds:
                seed_val += self.word_seeds[word]
            else:
                # Use first character value for unknown words
                seed_val += ord(word[0]) if word else 0
        
        random.seed(seed_val)
        
        # Generate a simple embedding
        embedding = np.array([random.uniform(-1, 1) for _ in range(self.dimension)])
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding


class StreamingChunker:
    """Process documents in a streaming fashion without loading entire document into memory"""
    
    def __init__(self, chunk_size=200, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text_stream(self, text):
        """Yield chunks from text without storing all chunks in memory"""
        if not text:
            return
            
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Try to find a good breaking point
            if end < text_length and end > start:
                # Look for period, question mark, or space in the last 30 chars of the current chunk
                last_30_chars = text[max(end-30, start):end]
                
                # Try to find sentence boundaries first
                sentence_end = max(
                    last_30_chars.rfind('.'), 
                    last_30_chars.rfind('?'),
                    last_30_chars.rfind('!')
                )
                
                if sentence_end != -1:
                    # Found a sentence boundary, adjust the end position
                    end = end - (30 - sentence_end) + 1
                else:
                    # Try to find a space as fallback
                    space = last_30_chars.rfind(' ')
                    if space != -1:
                        end = end - (30 - space) + 1
            
            # Extract just this chunk
            yield text[start:end]
            
            # Move the start position
            start = end - self.chunk_overlap
            
            # Force garbage collection to ensure memory is freed
            gc.collect()


class MinimalVectorStore:
    """Ultra-lightweight vector store that processes documents in batches"""
    
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self.max_chunks = 1000  # Limit total chunks to prevent memory issues
    
    def add_document_stream(self, chunker, text, source="unknown"):
        """Process a document chunk by chunk to minimize memory usage"""
        chunk_count = 0
        
        # Process document in a streaming fashion
        for chunk_text in chunker.chunk_text_stream(text):
            # Skip empty chunks
            if not chunk_text.strip():
                continue
                
            # Skip if we've reached capacity
            if len(self.chunks) >= self.max_chunks:
                print(f"Warning: Maximum chunks ({self.max_chunks}) reached. Some content will be skipped.")
                break
                
            # Add chunk
            self.chunks.append(chunk_text)
            
            # Add metadata
            self.metadata.append({
                "source": source,
                "chunk_index": chunk_count
            })
            
            # Create embedding for this chunk
            embedding = self.embedder.embed_text(chunk_text)
            self.embeddings.append(embedding)
            
            chunk_count += 1
            
            # Force garbage collection
            if chunk_count % 10 == 0:
                gc.collect()
                
        return chunk_count
    
    def search(self, query, top_k=3):
        """Find most relevant chunks for a query"""
        if not self.chunks:
            return []
            
        # Create query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Calculate similarities
        scores = []
        for doc_embedding in self.embeddings:
            # Use dot product for similarity
            similarity = np.dot(query_embedding, doc_embedding)
            scores.append(similarity)
        
        # Get top matches
        if not scores:
            return []
            
        # Find indices of top_k highest scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": scores[idx]
            })
            
        return results


class MinimalRAG:
    """Ultra-lightweight RAG system with streaming processing"""
    
    def __init__(self):
        # Initialize with minimal parameters for low memory usage
        self.embedder = MinimalEmbedder(dimension=16)
        self.chunker = StreamingChunker(chunk_size=150, chunk_overlap=15)
        self.vector_store = MinimalVectorStore(self.embedder)
    
    def add_document(self, content, source="unknown"):
        """Process and add a document in a streaming manner"""
        print(f"Processing document from {source}...")
        
        try:
            # Add document to vector store with streaming to minimize memory usage
            chunk_count = self.vector_store.add_document_stream(self.chunker, content, source)
            
            # Force garbage collection
            gc.collect()
            
            return f"Added document from '{source}' - processed {chunk_count} chunks"
            
        except MemoryError as e:
            gc.collect()  # Try to recover memory
            return f"Memory error: {e}. Try with shorter text or increase system memory."
        except Exception as e:
            return f"Error processing document: {e}"
    
    def query(self, query, top_k=3):
        """Process a query and generate a response"""
        try:
            # Find relevant chunks
            results = self.vector_store.search(query, top_k=top_k)
            
            if not results:
                return "No relevant information found."
            
            # Generate simple response (no LLM used here for memory efficiency)
            response = f"Results for query: '{query}'\n\n"
            
            for i, result in enumerate(results):
                response += f"[{i+1}] From {result['metadata']['source']} (Score: {result['score']:.2f})\n"
                response += f"{result['content']}\n\n"
                
            return response
            
        except MemoryError:
            gc.collect()  # Try to recover memory
            return "Memory error while processing query. Try a shorter query or increase system memory."
        except Exception as e:
            return f"Error processing query: {e}"


# Example usage - extremely minimal version
if __name__ == "__main__":
    import time
    
    # Track memory usage before starting
    print("Initializing minimal RAG system...")
    start_time = time.time()
    
    # Create RAG system
    rag = MinimalRAG()
    
    # Add very small sample documents for testing
    print("Adding sample documents...")
    
    try:
        doc1 = "RAG stands for Retrieval-Augmented Generation. It helps make AI responses more accurate."
        rag.add_document(doc1, source="RAG Definition")
        
        doc2 = "RAG systems combine search with text generation. They retrieve content and then use it."
        rag.add_document(doc2, source="RAG Process")
        
        doc3 = "Vector databases store embeddings that represent document meaning."
        rag.add_document(doc3, source="Vector Storage")
        
        # Test query
        query = "What is RAG?"
        print(f"\nQuerying: '{query}'")
        response = rag.query(query, top_k=2)
        
        print(f"\nResponse:\n{response}")
        
        # Report time
        end_time = time.time()
        print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in example: {e}")