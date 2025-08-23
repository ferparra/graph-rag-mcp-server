from __future__ import annotations
import os
import dspy
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Union, Any
from .config import settings
from .chroma_store import ChromaStore
from .graph_store import RDFGraphStore

class VaultQA(dspy.Signature):
    """Answer user questions grounded STRICTLY in provided vault snippets."""
    question = dspy.InputField()
    context = dspy.InputField(desc="relevant snippets from the vault")
    answer = dspy.OutputField(desc="concise answer with inline citations like [#] per snippet order")

class GeminiLM(dspy.LM):
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        self.client = genai.Client(api_key=self.api_key)
        super().__init__(model)
    
    def basic_request(self, prompt: str, **kwargs) -> List[Dict[str, str]]:
        try:
            generation_config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.1),
                max_output_tokens=kwargs.get("max_tokens", 900),
                top_p=kwargs.get("top_p", 0.9)
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=generation_config
            )
            
            if response and response.text:
                return [{"text": response.text}]
            else:
                return [{"text": "No response generated."}]
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return [{"text": f"Error: {str(e)}"}]

class ChromaRetriever:
    def __init__(self, store: ChromaStore, k: int = 6) -> None:
        self.store = store
        self.k = k
    
    def retrieve(self, query: str, where: Optional[Dict] = None) -> List[Dict]:
        return self.store.query(query, k=self.k, where=where)

class SemanticRetriever:
    """Enhanced retriever that uses both vector search and graph traversal."""
    
    def __init__(self, chroma_store: ChromaStore, graph_store: RDFGraphStore, k: int = 6) -> None:
        self.chroma_store = chroma_store
        self.graph_store = graph_store
        self.k = k
    
    def retrieve(self, query: str, where: Optional[Dict] = None, 
                expand_graph: bool = True, importance_threshold: float = 0.5) -> List[Dict]:
        """Retrieve using hybrid vector + graph approach."""
        
        # Step 1: Get initial chunks via vector similarity
        initial_hits = self.chroma_store.query(query, k=self.k, where=where)
        
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
            
            # Get neighboring chunks (sequential and hierarchical)
            if chunk_meta.get("semantic_chunk", False):
                neighbors = self.graph_store.get_chunk_neighbors(chunk_id)
                
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
                # This is a graph-expanded chunk - need to get content from ChromaDB
                try:
                    # Query ChromaDB for this specific chunk
                    chunk_hits = self.chroma_store.query(
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
        self.retriever = retriever  # Can be ChromaRetriever or SemanticRetriever
        self.model = model
        self.predict: Optional[dspy.Predict] = None
        
        try:
            self.lm = GeminiLM(model=model)
            dspy.configure(lm=self.lm)
            self.predict = dspy.Predict(VaultQA)
        except Exception as e:
            print(f"Warning: Could not initialize Gemini LM: {e}")
            self.predict = None
    
    def _call_gemini_direct(self, prompt: str) -> str:
        """Fallback method using direct Gemini API calls"""
        try:
            api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return "Error: Gemini API key not configured"
            
            client = genai.Client(api_key=api_key)
            
            generation_config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=900
            )
            
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=generation_config
            )
            
            return response.text if response and response.text else "No response generated."
            
        except Exception as e:
            return f"Error calling Gemini: {str(e)}"
    
    def forward(self, question: str, where: Optional[Dict] = None) -> dspy.Prediction:
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
        
        ctx = "\n".join(ctx_parts) if ctx_parts else "NO CONTEXT FOUND"
        
        if self.predict is not None:
            try:
                prediction = self.predict(question=question, context=ctx)
                # DSPy Predict.__call__ should return dspy.Prediction synchronously
                # Type check to ensure we have the expected type
                if isinstance(prediction, dspy.Prediction):
                    return prediction
                elif hasattr(prediction, '__await__'):
                    print("DSPy returned coroutine in sync context, falling back to direct API")
                else:
                    print(f"DSPy returned unexpected type {type(prediction)}, falling back to direct API")
            except Exception as e:
                print(f"DSPy prediction error: {e}, falling back to direct API")
        
        prompt = f"""You are a strict RAG assistant for an Obsidian vault. Use ONLY the provided snippets to answer questions.

Snippets:
{ctx}

Question: {question}

Rules:
- Answer concisely and accurately
- Use ONLY information from the provided snippets
- If insufficient evidence, say "I don't have enough information in the vault to answer this question"
- Add bracketed citations [#] referring to the snippet numbers used
- Focus on being helpful while staying grounded in the vault content

Answer:"""
        
        answer = self._call_gemini_direct(prompt)
        
        return dspy.Prediction(
            question=question,
            context=ctx,
            answer=answer
        )

def build_rag(graph_store=None) -> RAGProgram:
    """Build and return a RAG program with appropriate retriever based on configuration."""
    chroma_store = ChromaStore(
        client_dir=settings.chroma_dir,
        collection_name=settings.collection,
        embed_model=settings.embedding_model
    )
    
    if settings.chunk_strategy == "semantic" and graph_store is not None:
        # Use semantic retriever with graph expansion
        retriever = SemanticRetriever(
            chroma_store=chroma_store,
            graph_store=graph_store,
            k=6
        )
    else:
        # Use traditional vector-only retriever
        retriever = ChromaRetriever(chroma_store, k=6)
    
    return RAGProgram(retriever, model=settings.gemini_model)

class VaultSearcher:
    """Higher-level interface for vault search and Q&A"""
    
    def __init__(self, graph_store=None):
        self.graph_store = graph_store
        self.rag = build_rag(graph_store=graph_store)
        self.chroma_store = ChromaStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
    
    def search(self, query: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Search vault for relevant snippets"""
        return self.chroma_store.query(query, k=k, where=where)
    
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
