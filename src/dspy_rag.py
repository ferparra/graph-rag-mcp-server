from __future__ import annotations
import os
import dspy
from typing import List, Dict, Optional
from .config import settings
from .unified_store import UnifiedStore

class VaultQA(dspy.Signature):
    """Answer user questions grounded STRICTLY in provided vault snippets."""
    context = dspy.InputField(desc="relevant snippets from the vault")
    question = dspy.InputField()
    response = dspy.OutputField(desc="concise answer with inline citations like [#] per snippet order")



class UnifiedRetrieverCompat:
    """Compatibility wrapper for unified store as a simple retriever."""
    
    def __init__(self, store: UnifiedStore, k: int = 6) -> None:
        self.store = store
        self.k = k
    
    def retrieve(self, query: str, where: Optional[Dict] = None) -> List[Dict]:
        return self.store.query(query, k=self.k, where=where)

class UnifiedRetriever:
    """Enhanced retriever using unified ChromaDB store with graph capabilities."""
    
    def __init__(self, unified_store: UnifiedStore, k: int = 6) -> None:
        self.unified_store = unified_store
        self.k = k
    
    def retrieve(self, query: str, where: Optional[Dict] = None, 
                expand_graph: bool = True, importance_threshold: float = 0.5) -> List[Dict]:
        """Retrieve using hybrid vector + graph approach with unified store."""
        
        # Step 1: Get initial chunks via vector similarity
        initial_hits = self.unified_store.query(query, k=self.k, where=where)
        
        if not expand_graph or settings.chunk_strategy != "semantic":
            return initial_hits
        
        # Step 2: Expand with graph-connected chunks
        expanded_chunks = set()
        chunk_scores = {}
        
        for hit in initial_hits:
            chunk_id = hit.get("id", "")
            chunk_meta = hit.get("meta", {})
            
            # Store original vector score
            vector_score = 1.0 - hit.get("distance", 0.0)  # Convert distance to similarity
            chunk_scores[chunk_id] = {
                "vector_score": vector_score,
                "importance_score": chunk_meta.get("importance_score", 0.5),
                "chunk_type": chunk_meta.get("chunk_type", "unknown"),
                "header_text": chunk_meta.get("header_text"),
                "original_hit": hit
            }
            expanded_chunks.add(chunk_id)
            
            # Get neighboring chunks (sequential and hierarchical) using unified store
            if chunk_meta.get("semantic_chunk", False):
                neighbors = self.unified_store.get_chunk_neighbors(chunk_id)
                
                for neighbor in neighbors:
                    neighbor_id = neighbor["chunk_id"]
                    if (neighbor_id not in expanded_chunks and 
                        neighbor["importance_score"] >= importance_threshold):
                        
                        # Calculate composite score for neighbors
                        neighbor_score = (
                            vector_score * 0.3 +  # Inherited vector relevance
                            neighbor["importance_score"] * 0.7  # Chunk importance
                        )
                        
                        chunk_scores[neighbor_id] = {
                            "vector_score": vector_score * 0.3,  # Inherited
                            "importance_score": neighbor["importance_score"],
                            "chunk_type": neighbor["chunk_type"],
                            "header_text": neighbor.get("header"),
                            "relationship": neighbor["relationship"],
                            "composite_score": neighbor_score
                        }
                        expanded_chunks.add(neighbor_id)
        
        # Step 3: Re-rank all chunks by composite score
        ranked_chunks = []
        
        for chunk_id in expanded_chunks:
            score_info = chunk_scores[chunk_id]
            
            if "original_hit" in score_info:
                # This was an original vector search hit
                chunk_data = score_info["original_hit"]
                chunk_data["composite_score"] = (
                    score_info["vector_score"] * 0.7 + 
                    score_info["importance_score"] * 0.3
                )
                chunk_data["retrieval_method"] = "vector_search"
            else:
                # This is a graph-expanded chunk - get content from unified store
                try:
                    # Query unified store for this specific chunk
                    chunk_hits = self.unified_store.query(
                        f"chunk_id:{chunk_id}", k=1, 
                        where={"chunk_id": {"$eq": chunk_id}}
                    )
                    
                    if chunk_hits:
                        chunk_data = chunk_hits[0]
                        chunk_data["composite_score"] = score_info.get("composite_score", 0.5)
                        chunk_data["retrieval_method"] = "graph_expansion"
                        chunk_data["relationship"] = score_info.get("relationship", "unknown")
                    else:
                        # Fallback: create minimal chunk data
                        chunk_data = {
                            "id": chunk_id,
                            "text": f"[Chunk content not found for {chunk_id}]",
                            "meta": {
                                "chunk_type": score_info["chunk_type"],
                                "header_text": score_info["header_text"],
                                "importance_score": score_info["importance_score"]
                            },
                            "composite_score": score_info.get("composite_score", 0.5),
                            "retrieval_method": "graph_expansion",
                            "relationship": score_info.get("relationship", "unknown")
                        }
                except Exception as e:
                    print(f"Error retrieving chunk {chunk_id}: {e}")
                    continue
            
            ranked_chunks.append(chunk_data)
        
        # Sort by composite score and return top k results
        ranked_chunks.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return ranked_chunks[:self.k * 2]  # Return more results due to expansion

class RAGProgram(dspy.Module):
    def __init__(self, retriever, model: str = "gemini-2.5-flash"):
        super().__init__()
        self.retriever = retriever  # Can be UnifiedRetrieverCompat or UnifiedRetriever
        self.model = model
        self.rag: Optional[dspy.ChainOfThought] = None
        
        try:
            api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            
            # Use DSPy's built-in Gemini support
            self.lm = dspy.LM(f"gemini/{model}", api_key=api_key)
            dspy.configure(lm=self.lm)
            
            # Use DSPy's ChainOfThought for RAG
            self.rag = dspy.ChainOfThought(VaultQA)
        except Exception as e:
            print(f"Warning: Could not initialize Gemini LM: {e}")
            self.rag = None
    
    def _search_vault(self, question: str, where: Optional[Dict] = None) -> str:
        """Search vault and return formatted context snippets."""
        hits = self.retriever.retrieve(question, where=where)
        
        # Create enhanced context with semantic information
        ctx_parts = []
        for i, h in enumerate(hits, start=1):
            title = h['meta'].get('title', 'Unknown')
            path = h['meta'].get('path', 'Unknown')
            text = h['text']
            
            # Add semantic chunk information if available
            chunk_meta = h.get('meta', {})
            semantic_info = ""
            
            if chunk_meta.get('semantic_chunk', False):
                chunk_type = chunk_meta.get('chunk_type', '')
                header_text = chunk_meta.get('header_text', '')
                importance = chunk_meta.get('importance_score', 0.5)
                retrieval_method = h.get('retrieval_method', 'vector_search')
                
                if header_text:
                    semantic_info = f" [{header_text}]"
                elif chunk_meta.get('parent_headers'):
                    parent_headers = chunk_meta['parent_headers']
                    if parent_headers:
                        semantic_info = f" [{' > '.join(parent_headers[-2:])}]"
                
                semantic_info += f" ({chunk_type}, score:{importance:.2f}, via:{retrieval_method})"
            
            ctx_parts.append(f"[{i}] {title}{semantic_info} ({path})\n{text}\n")
        
        return "\n".join(ctx_parts) if ctx_parts else "NO CONTEXT FOUND"
    
    def forward(self, question: str, where: Optional[Dict] = None) -> dspy.Prediction:
        # Search vault for relevant context
        context = self._search_vault(question, where=where)
        
        if self.rag is not None:
            try:
                # Use DSPy's ChainOfThought for RAG
                prediction = self.rag.forward(context=context, question=question)
                
                # Ensure we return a proper Prediction with expected fields
                if isinstance(prediction, dspy.Prediction):
                    # Add context field for compatibility
                    if not hasattr(prediction, 'context'):
                        prediction.context = context
                    # Map response to answer for compatibility
                    if hasattr(prediction, 'response') and not hasattr(prediction, 'answer'):
                        prediction.answer = prediction.response
                    return prediction
                    
            except Exception as e:
                print(f"DSPy ChainOfThought error: {e}, using fallback")
        
        # Fallback: create manual prediction
        return dspy.Prediction(
            question=question,
            context=context,
            answer="I don't have enough information in the vault to answer this question",
            response="I don't have enough information in the vault to answer this question"
        )

def build_rag(unified_store: UnifiedStore) -> RAGProgram:
    """Build and return a RAG program with DSPy ChainOfThought and appropriate retriever."""
    
    # Use unified store with graph capabilities
    if settings.chunk_strategy == "semantic":
        retriever = UnifiedRetriever(unified_store=unified_store, k=6)
    else:
        retriever = UnifiedRetrieverCompat(unified_store, k=6)
    
    return RAGProgram(retriever, model=settings.gemini_model)

class VaultSearcher:
    """Higher-level interface for vault search and Q&A"""
    
    def __init__(self, unified_store: UnifiedStore):
        self.unified_store = unified_store
        self.rag = build_rag(unified_store=unified_store)
        self.search_store = unified_store
    
    def search(self, query: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Search vault for relevant snippets"""
        return self.search_store.query(query, k=k, where=where)
    
    def ask(self, question: str, where: Optional[Dict] = None) -> Dict:
        """Ask a question and get a RAG-powered answer"""
        try:
            prediction = self.rag.forward(question=question, where=where)
            
            return {
                "question": question,
                "answer": getattr(prediction, 'answer', 'No answer generated'),
                "context": getattr(prediction, 'context', 'No context available'),
                "success": True
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "context": "",
                "success": False
            }
