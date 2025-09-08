from __future__ import annotations
import os
from pathlib import Path

# Ensure DSPy and related caches write to a workspace-writable location before importing dspy
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(Path.cwd() / ".cache")
# Ensure DSPy uses a writable cache directory in this workspace
if "DSPY_CACHEDIR" not in os.environ:
    os.environ["DSPY_CACHEDIR"] = str(Path(os.environ["XDG_CACHE_HOME"]) / "dspy")

import dspy
from typing import List, Dict, Optional, Any
import logging

# Support both package and module execution contexts
try:
    from config import settings
    from unified_store import UnifiedStore
    from dspy_programs import AdaptiveRAGProgram
    from dspy_optimizer import OptimizationManager
    from dspy_agent import ComplexQueryHandler
    from dspy_signatures import VaultQA
except ImportError:  # When imported as part of a package
    from .config import settings
    from .unified_store import UnifiedStore
    from .dspy_optimizer import OptimizationManager
    from .dspy_agent import ComplexQueryHandler
    from .dspy_signatures import VaultQA

logger = logging.getLogger(__name__)

# VaultQA signature imported from dspy_signatures module



class UnifiedRetrieverCompat:
    """Compatibility wrapper for unified store as a simple retriever."""
    
    def __init__(self, store: UnifiedStore, k: int = 6) -> None:
        self.store = store
        self.k = k
    
    def retrieve(self, query: str, where: Optional[Dict] = None) -> List[Dict]:
        return self.store.query(query, k=self.k, where=where)

class UnifiedRetriever:
    """Enhanced retriever using unified ChromaDB store with intelligent graph expansion."""
    
    def __init__(self, unified_store: UnifiedStore, k: int = 6) -> None:
        self.unified_store = unified_store
        self.k = k
    
    def retrieve(self, query: str, where: Optional[Dict] = None, 
                expand_graph: bool = True, importance_threshold: float = 0.3,
                diversity_threshold: float = 0.7) -> List[Dict]:
        """Retrieve using intelligent hybrid vector + graph approach."""
        
        # Step 1: Get initial chunks via vector similarity
        initial_hits = self.unified_store.query(query, k=min(self.k * 2, 12), where=where)
        
        if not expand_graph or settings.chunk_strategy != "semantic":
            return initial_hits[:self.k]
        
        # Step 2: Analyze initial hits for expansion strategy
        expansion_candidates = self._select_expansion_candidates(initial_hits, query)
        
        # Step 3: Multi-level graph expansion
        expanded_chunks = {}
        relationship_scores = {}
        
        for candidate in expansion_candidates:
            chunk_id = candidate.get("id", "")
            if not chunk_id:
                continue
                
            candidate_score = 1.0 - candidate.get("distance", 0.0)
            expanded_chunks[chunk_id] = {
                "chunk_data": candidate,
                "vector_score": candidate_score,
                "expansion_level": 0,
                "source_chunk": chunk_id
            }
            
            # Expand from this candidate
            self._expand_from_chunk(chunk_id, candidate_score, expanded_chunks, 
                                  relationship_scores, importance_threshold, max_depth=2)
        
        # Step 4: Intelligent re-ranking with diversity
        final_chunks = self._intelligent_rerank(
            expanded_chunks, relationship_scores, query, diversity_threshold
        )
        
        return final_chunks[:self.k]
    
    def _select_expansion_candidates(self, initial_hits: List[Dict], query: str) -> List[Dict]:
        """Select best candidates for graph expansion based on relevance and chunk quality."""
        if not initial_hits:
            return []
        
        candidates = []
        query_lower = query.lower()
        
        for hit in initial_hits:
            meta = hit.get("meta", {})
            
            # Base relevance score
            relevance_score = 1.0 - hit.get("distance", 0.0)
            
            # Boost for high-importance chunks
            importance = meta.get("importance_score", 0.5)
            if importance > 0.7:
                relevance_score += 0.1
            
            # Boost for header chunks (more likely to have good connections)
            chunk_type = meta.get("chunk_type", "")
            if chunk_type in ["section", "header"]:
                relevance_score += 0.05
            
            # Boost for chunks with links (more connected)
            links = meta.get("contains_links", "")
            if links:
                link_count = len(links.split(",")) if isinstance(links, str) else 0
                relevance_score += min(link_count * 0.02, 0.1)
            
            # Boost for query term overlap in headers
            header_text = meta.get("header_text", "").lower()
            if header_text and any(term in header_text for term in query_lower.split()):
                relevance_score += 0.08
            
            hit["expansion_score"] = relevance_score
            candidates.append(hit)
        
        # Sort by expansion score and take top candidates
        candidates.sort(key=lambda x: x.get("expansion_score", 0), reverse=True)
        return candidates[:max(3, self.k // 2)]  # Limit expansion sources
    
    def _expand_from_chunk(self, source_chunk_id: str, source_score: float,
                          expanded_chunks: Dict, relationship_scores: Dict,
                          importance_threshold: float, max_depth: int = 2) -> None:
        """Recursively expand from a chunk through graph relationships."""
        
        current_level = [(source_chunk_id, source_score, 0)]
        
        for _ in range(max_depth):
            next_level = []
            
            for chunk_id, inherited_score, current_depth in current_level:
                if current_depth >= max_depth:
                    continue
                
                # Get neighbors with different relationship priorities
                neighbors = self.unified_store.get_chunk_neighbors(
                    chunk_id, include_sequential=True, include_hierarchical=True
                )
                
                for neighbor in neighbors:
                    neighbor_id = neighbor["chunk_id"]
                    relationship = neighbor["relationship"]
                    neighbor_importance = neighbor.get("importance_score", 0.5)
                    
                    # Skip if already processed or below threshold
                    if (neighbor_id in expanded_chunks or 
                        neighbor_importance < importance_threshold):
                        continue
                    
                    # Calculate relationship-aware score
                    relationship_weight = self._get_relationship_weight(relationship, current_depth)
                    composite_score = (
                        inherited_score * 0.4 +  # Inherited relevance
                        neighbor_importance * 0.4 +  # Chunk quality
                        relationship_weight * 0.2  # Relationship strength
                    )
                    
                    # Only include if score is meaningful
                    if composite_score > 0.3:
                        # Get the actual chunk content
                        try:
                            chunk_hits = self.unified_store._collection().get(
                                where={"chunk_id": {"$eq": neighbor_id}},
                                include=['metadatas', 'documents']
                            )
                            
                            metadatas = chunk_hits.get('metadatas')
                            if metadatas and len(metadatas) > 0:
                                meta = metadatas[0]
                                documents = chunk_hits.get('documents')
                                doc = (documents[0] if documents and len(documents) > 0 else "")
                                
                                chunk_data = {
                                    "id": neighbor_id,
                                    "text": doc,
                                    "meta": meta,
                                    "distance": 1.0 - composite_score
                                }
                                
                                expanded_chunks[neighbor_id] = {
                                    "chunk_data": chunk_data,
                                    "vector_score": inherited_score * 0.4,
                                    "expansion_level": current_depth + 1,
                                    "source_chunk": source_chunk_id,
                                    "relationship": relationship,
                                    "composite_score": composite_score
                                }
                                
                                # Track relationship for diversity calculation
                                rel_key = f"{source_chunk_id}->{neighbor_id}"
                                relationship_scores[rel_key] = {
                                    "type": relationship,
                                    "strength": relationship_weight,
                                    "depth": current_depth + 1
                                }
                                
                                # Add to next level for further expansion
                                if current_depth + 1 < max_depth:
                                    next_level.append((neighbor_id, composite_score, current_depth + 1))
                        
                        except Exception as e:
                            print(f"Error expanding from chunk {neighbor_id}: {e}")
                            continue
            
            current_level = next_level
    
    def _get_relationship_weight(self, relationship: str, depth: int) -> float:
        """Get weight for different relationship types, adjusted by depth."""
        base_weights = {
            "sequential_next": 0.8,
            "sequential_prev": 0.8,
            "parent": 0.9,
            "child": 0.7,
            "sibling": 0.6,
            "content_link": 0.9
        }
        
        weight = base_weights.get(relationship, 0.5)
        
        # Decay weight by depth
        depth_factor = 0.8 ** depth
        return weight * depth_factor
    
    def _intelligent_rerank(self, expanded_chunks: Dict, relationship_scores: Dict,
                           query: str, diversity_threshold: float) -> List[Dict]:
        """Intelligent re-ranking with diversity and relevance balancing."""
        
        if not expanded_chunks:
            return []
        
        # Convert to list for processing
        candidates = []
        for chunk_id, chunk_info in expanded_chunks.items():
            chunk_data = chunk_info["chunk_data"]
            
            # Calculate final score
            final_score = chunk_info.get("composite_score", chunk_info["vector_score"])
            
            # Boost for direct vector hits
            if chunk_info["expansion_level"] == 0:
                final_score *= 1.2
            
            # Add query-specific relevance boost
            text_relevance = self._calculate_text_relevance(chunk_data.get("text", ""), query)
            final_score += text_relevance * 0.1
            
            chunk_data["final_score"] = final_score
            chunk_data["retrieval_method"] = (
                "vector_search" if chunk_info["expansion_level"] == 0 
                else "graph_expansion"
            )
            chunk_data["expansion_info"] = {
                "level": chunk_info["expansion_level"],
                "source": chunk_info["source_chunk"],
                "relationship": chunk_info.get("relationship", "direct")
            }
            
            candidates.append(chunk_data)
        
        # Sort by final score
        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        # Apply diversity filtering to avoid too many similar chunks
        diverse_results = self._apply_diversity_filter(candidates, diversity_threshold)
        
        return diverse_results
    
    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """Calculate text-based relevance score."""
        if not text or not query:
            return 0.0
        
        text_lower = text.lower()
        query_terms = [term.strip() for term in query.lower().split() if len(term.strip()) > 2]
        
        if not query_terms:
            return 0.0
        
        # Simple term frequency scoring
        total_matches = 0
        for term in query_terms:
            total_matches += text_lower.count(term)
        
        # Normalize by text length and query terms
        relevance = total_matches / (len(text) / 100 + len(query_terms))
        return min(relevance, 1.0)
    
    def _apply_diversity_filter(self, candidates: List[Dict], threshold: float) -> List[Dict]:
        """Filter results to ensure diversity while maintaining relevance."""
        if len(candidates) <= self.k:
            return candidates
        
        selected = []
        selected.append(candidates[0])  # Always include top result
        
        for candidate in candidates[1:]:
            if len(selected) >= self.k:
                break
            
            # Check diversity against already selected chunks
            is_diverse = True
            candidate_meta = candidate.get("meta", {})
            candidate_note = candidate_meta.get("note_id", "")
            candidate_type = candidate_meta.get("chunk_type", "")
            
            for selected_chunk in selected:
                selected_meta = selected_chunk.get("meta", {})
                selected_note = selected_meta.get("note_id", "")
                selected_type = selected_meta.get("chunk_type", "")
                
                # Same note and chunk type = low diversity
                if (candidate_note == selected_note and 
                    candidate_type == selected_type and
                    candidate.get("final_score", 0) < threshold):
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(candidate)
        
        return selected

class RAGProgram(dspy.Module):
    """Legacy RAG program for backward compatibility. Use EnhancedVaultSearcher for new features."""
    
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
            logger.warning(f"Could not initialize Gemini LM: {e}")
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

class EnhancedVaultSearcher:
    """Enhanced vault searcher with DSPy optimization and ReAct agent capabilities."""
    
    def __init__(self, unified_store: UnifiedStore):
        self.unified_store = unified_store
        self.search_store = unified_store
        
        # Initialize optimization manager if enabled
        self.optimization_manager: Optional[OptimizationManager] = None
        self.adaptive_rag: Optional[AdaptiveRAGProgram] = None
        
        if settings.dspy_optimize_enabled:
            try:
                self.optimization_manager = OptimizationManager(unified_store)
                self.adaptive_rag = self.optimization_manager.get_program()
                logger.info("Enhanced RAG with optimization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize optimization: {e}")
                self.optimization_manager = None
                self.adaptive_rag = None
        
        # Initialize complex query handler
        self.complex_handler: Optional[ComplexQueryHandler] = None
        try:
            self.complex_handler = ComplexQueryHandler(
                unified_store, 
                settings.dspy_state_dir if settings.dspy_optimize_enabled else None
            )
        except Exception as e:
            logger.warning(f"Failed to initialize complex query handler: {e}")
            self.complex_handler = None
        
        # Fallback to legacy RAG
        self.legacy_rag = build_rag(unified_store=unified_store)
    
    def search(self, query: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Search vault for relevant snippets"""
        return self.search_store.query(query, k=k, where=where)
    
    async def ask(self, question: str, where: Optional[Dict] = None, use_enhanced: bool = True) -> Dict:
        """Ask a question and get an optimized RAG-powered answer"""
        
        # Try enhanced pipeline first
        if use_enhanced and self.adaptive_rag:
            try:
                logger.info("Using enhanced adaptive RAG")
                prediction = self.adaptive_rag.forward(question=question)
                
                return {
                    "question": question,
                    "answer": getattr(prediction, 'answer', 'No answer generated'),
                    "context": getattr(prediction, 'context', 'No context available'),
                    "method": "enhanced_adaptive_rag",
                    "query_intent": getattr(prediction, 'query_intent', 'unknown'),
                    "intent_confidence": getattr(prediction, 'intent_confidence', 'unknown'),
                    "retrieval_method": getattr(prediction, 'retrieval_method', 'unknown'),
                    "citations": getattr(prediction, 'citations', []),
                    "confidence": getattr(prediction, 'confidence', 'unknown'),
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Enhanced RAG failed: {e}")
                # Fall through to other methods
        
        # Try complex query handler for multi-hop questions
        if self.complex_handler and self.complex_handler.should_use_agent(question):
            try:
                logger.info("Using complex query handler")
                result = await self.complex_handler.handle_query(question)
                if result.get("success"):
                    result["method"] = "complex_agent"
                    return result
            except Exception as e:
                logger.error(f"Complex query handler failed: {e}")
        
        # Fallback to legacy RAG
        try:
            logger.info("Using legacy RAG")
            prediction = self.legacy_rag.forward(question=question, where=where)
            
            return {
                "question": question,
                "answer": getattr(prediction, 'answer', 'No answer generated'),
                "context": getattr(prediction, 'context', 'No context available'),
                "method": "legacy_rag",
                "success": True
            }
        except Exception as e:
            logger.error(f"All RAG methods failed: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "context": "",
                "method": "error",
                "success": False,
                "error": str(e)
            }
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization of the enhanced RAG program."""
        if self.optimization_manager:
            return self.optimization_manager.force_optimization()
        else:
            return {
                "success": False,
                "message": "Optimization not enabled",
                "program_optimized": False
            }
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization status."""
        if self.optimization_manager:
            return self.optimization_manager.get_status()
        else:
            return {
                "optimization_enabled": False,
                "message": "Optimization not available"
            }


class VaultSearcher:
    """Legacy vault searcher for backward compatibility."""
    
    def __init__(self, unified_store: UnifiedStore):
        self.unified_store = unified_store
        self.enhanced_searcher = EnhancedVaultSearcher(unified_store)
        
        # Maintain legacy interface
        self.rag = self.enhanced_searcher.legacy_rag
        self.search_store = unified_store
    
    def search(self, query: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Search vault for relevant snippets"""
        return self.enhanced_searcher.search(query, k, where)
    
    async def ask(self, question: str, where: Optional[Dict] = None) -> Dict:
        """Ask a question and get a RAG-powered answer (uses enhanced searcher)"""
        return await self.enhanced_searcher.ask(question, where, use_enhanced=True)
