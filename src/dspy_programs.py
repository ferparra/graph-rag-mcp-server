"""
Enhanced DSPy Programs for Graph RAG System

This module implements sophisticated DSPy programs that leverage the full power of DSPy's
optimization capabilities while maintaining state persistence across uvx ephemeral connections.
"""

from __future__ import annotations
import dspy
from pathlib import Path
from typing import List, Dict, Optional, Any, cast
import json
import time
import logging
import inspect

# Support both package and module execution contexts
try:
    from dspy_signatures import (
        QueryIntentClassifier, AnswerWithCitations, ContextExpansion, 
        SemanticSimilarityJudge, ConceptSynthesis, ContentQualityAssessment
    )
    from unified_store import UnifiedStore
    from config import settings
except ImportError:  # When imported as part of a package
    from .dspy_signatures import (
        QueryIntentClassifier, AnswerWithCitations, ContextExpansion, SemanticSimilarityJudge,
        ConceptSynthesis, ContentQualityAssessment
    )
    from .unified_store import UnifiedStore
    from .config import settings

logger = logging.getLogger(__name__)


class StateManager:
    """Manages persistent state for DSPy programs across ephemeral connections."""
    
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_log = state_dir / "optimization_log.json"
        self.metrics_dir = state_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
    
    def save_program(self, program: dspy.Module, name: str) -> None:
        """Save a DSPy program's optimized state."""
        try:
            state_file = self.state_dir / f"{name}.json"
            program.save(str(state_file), save_program=False)
            logger.info(f"Saved optimized program '{name}' to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save program '{name}': {e}")
    
    def load_program(self, name: str) -> Optional[dspy.Module]:
        """Load a DSPy program's optimized state."""
        try:
            state_file = self.state_dir / f"{name}.json"
            if state_file.exists():
                program = dspy.load(str(state_file))
                logger.info(f"Loaded optimized program '{name}' from {state_file}")
                return program
        except Exception as e:
            logger.error(f"Failed to load program '{name}': {e}")
        return None
    
    def log_optimization(self, name: str, metrics: Dict[str, Any]) -> None:
        """Log optimization metrics."""
        try:
            log_entry = {
                "timestamp": time.time(),
                "program": name,
                "metrics": metrics
            }
            
            # Load existing log
            log_data = []
            if self.optimization_log.exists():
                with open(self.optimization_log, 'r') as f:
                    log_data = json.load(f)
            
            # Append new entry
            log_data.append(log_entry)
            
            # Keep only last 100 entries
            log_data = log_data[-100:]
            
            # Save updated log
            with open(self.optimization_log, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log optimization for '{name}': {e}")
    
    def should_optimize(self, name: str) -> bool:
        """Check if a program should be re-optimized based on staleness."""
        try:
            state_file = self.state_dir / f"{name}.json"
            if not state_file.exists():
                return True
            
            # Check file age
            file_age_hours = (time.time() - state_file.stat().st_mtime) / 3600
            return file_age_hours > settings.dspy_optimization_interval_hours
            
        except Exception:
            return True


class QueryRouter(dspy.Module):
    """Intelligent query router that analyzes intent and selects optimal strategy."""
    
    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        
        # Try to load optimized classifier
        optimized = state_manager.load_program("query_intent_classifier")
        if optimized:
            self.classifier = optimized
        else:
            self.classifier = dspy.ChainOfThought(QueryIntentClassifier)
    
    async def forward(self, query: str) -> dspy.Prediction:
        """Classify query intent and return routing decision."""
        try:
            # dspy modules are dynamically callable; use Any to satisfy type checker
            classifier_any: Any = self.classifier
            result_any: Any = classifier_any(query=query)

            # Handle potential async result with precise typing
            prediction_obj: Any
            if inspect.isawaitable(result_any):
                prediction_obj = await result_any
            else:
                prediction_obj = result_any

            # Coerce into a dspy.Prediction for type safety
            prediction: dspy.Prediction
            if isinstance(prediction_obj, dspy.Prediction):
                prediction = prediction_obj
            else:
                prediction = dspy.Prediction()

            # Ensure we have required fields
            if not hasattr(prediction, 'intent'):
                prediction.intent = "semantic"  # fallback
            if not hasattr(prediction, 'confidence'):
                prediction.confidence = "0.5"  # fallback
            if not hasattr(prediction, 'reasoning'):
                prediction.reasoning = "Default routing"

            return prediction
        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            # Return fallback prediction
            return dspy.Prediction(
                intent="semantic",
                confidence="0.5",
                reasoning="Fallback due to routing error"
            )


class AdaptiveRetriever(dspy.Module):
    """Multi-strategy retrieval with intelligent context expansion."""
    
    def __init__(self, unified_store: UnifiedStore, state_manager: StateManager):
        super().__init__()
        self.unified_store = unified_store
        self.state_manager = state_manager
        
        # Load optimized components
        # Use Any annotations to avoid strict callable typing issues from dspy dynamic modules
        self.similarity_judge: Any = (
            state_manager.load_program("similarity_judge") or
            dspy.ChainOfThought(SemanticSimilarityJudge)
        )
        self.context_expander: Any = (
            state_manager.load_program("context_expander") or
            dspy.ChainOfThought(ContextExpansion)
        )
        self.quality_assessor: Any = (
            state_manager.load_program("quality_assessor") or
            dspy.ChainOfThought(ContentQualityAssessment)
        )
    
    async def forward(self, query: str, intent: str = "semantic", k: int = 6,
                max_expansion_rounds: int = 2) -> dspy.Prediction:
        """Adaptive retrieval with context expansion."""
        
        # Initial retrieval based on intent
        if intent == "graph":
            initial_results = self._graph_retrieval(query, k)
        elif intent == "categorical":
            initial_results = self._categorical_retrieval(query, k)
        elif intent == "specific":
            initial_results = self._specific_retrieval(query, k)
        else:  # semantic or default
            initial_results = self._semantic_retrieval(query, k)
        
        # Assess quality and expand if needed
        context = self._format_context(initial_results)
        all_results = initial_results
        
        for round_num in range(max_expansion_rounds):
            try:
                assessor_any: Any = self.quality_assessor
                quality_result = assessor_any(
                    question=query,
                    retrieved_content=context
                )
                
                # Handle potential async result
                if inspect.isawaitable(quality_result):
                    quality_pred = await quality_result
                else:
                    quality_pred = quality_result
                
                # Parse completeness score
                completeness = self._parse_score(getattr(quality_pred, 'completeness_score', '0.0'))
                
                if completeness > 0.7:  # Good enough
                    break
                
                # Expand context
                expander_any: Any = self.context_expander
                expansion_result = expander_any(
                    current_context=context,
                    question=query
                )
                
                # Handle potential async result
                if inspect.isawaitable(expansion_result):
                    expansion_pred = await expansion_result
                else:
                    expansion_pred = expansion_result
                
                if str(getattr(expansion_pred, 'expansion_needed', '')).lower() == "yes":
                    expanded_results = self._expand_context(
                        str(getattr(expansion_pred, 'expansion_strategy', '')), k//2
                    )
                    all_results.extend(expanded_results)
                    context = self._format_context(all_results)
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Context expansion failed in round {round_num}: {e}")
                break
        
        return dspy.Prediction(
            results=all_results[:k*2],  # Limit total results
            context=context,
            retrieval_method=intent,
            expansion_rounds=round_num + 1
        )
    
    def _semantic_retrieval(self, query: str, k: int) -> List[Dict]:
        """Standard semantic similarity retrieval."""
        return self.unified_store.query(query, k=k)
    
    def _graph_retrieval(self, query: str, k: int) -> List[Dict]:
        """Graph-based retrieval focusing on relationships."""
        # First get initial hits
        initial = self.unified_store.query(query, k=k//2)
        
        # Expand through graph relationships
        expanded = []
        for hit in initial:
            chunk_id = hit.get("id", "")
            if chunk_id:
                neighbors = self.unified_store.get_chunk_neighbors(
                    chunk_id, include_sequential=True, include_hierarchical=True
                )
                for neighbor in neighbors[:2]:  # Limit expansion
                    try:
                        chunk_data = self.unified_store._collection().get(
                            where={"chunk_id": {"$eq": neighbor["chunk_id"]}},
                            include=['metadatas', 'documents']
                        )
                        if isinstance(chunk_data, dict):
                            documents = chunk_data.get('documents') or []
                            metadatas = chunk_data.get('metadatas') or []
                            if isinstance(documents, list) and documents:
                                expanded.append({
                                    "id": neighbor["chunk_id"],
                                    "text": documents[0],
                                    "meta": metadatas[0] if isinstance(metadatas, list) and metadatas else {},
                                    "distance": 0.5,  # Graph-based
                                    "retrieval_method": "graph_expansion"
                                })
                    except Exception as e:
                        logger.error(f"Graph expansion error: {e}")
        
        # Combine and deduplicate
        all_results = initial + expanded
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = result.get("id", "")
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        return unique_results[:k]
    
    def _categorical_retrieval(self, query: str, k: int) -> List[Dict]:
        """Tag-based categorical retrieval."""
        # Extract potential tags from query
        import re
        tag_patterns = re.findall(r'#(\w+)|tag[s]?\s*[:=]\s*(\w+)', query.lower())
        tags = [tag for match in tag_patterns for tag in match if tag]
        
        if tags:
            # Use tag-based filtering
            where_filter = {"tags": {"$in": tags}}
            return self.unified_store.query(query, k=k, where=where_filter)
        else:
            # Fallback to semantic
            return self._semantic_retrieval(query, k)
    
    def _specific_retrieval(self, query: str, k: int) -> List[Dict]:
        """Targeted retrieval for specific information."""
        # Use smaller k for more focused results
        return self.unified_store.query(query, k=min(k, 4))
    
    def _expand_context(self, strategy: str, k: int) -> List[Dict]:
        """Expand context based on strategy suggestion."""
        try:
            # Parse strategy for search terms
            if "search" in strategy.lower():
                # Extract quoted terms or key phrases
                import re
                terms = re.findall(r'"([^"]*)"|\b(\w+(?:\s+\w+){0,2})\b', strategy)
                search_terms = [term for match in terms for term in match if term]
                
                if search_terms:
                    return self.unified_store.query(" ".join(search_terms[:3]), k=k)
            
            # Fallback: use original strategy text as query
            return self.unified_store.query(strategy, k=k)
            
        except Exception as e:
            logger.error(f"Context expansion failed: {e}")
            return []
    
    def _format_context(self, results: List[Dict]) -> str:
        """Format results into context string."""
        if not results:
            return "NO CONTEXT AVAILABLE"
        
        context_parts: List[str] = []
        for i, result in enumerate(results, 1):
            title = result.get('meta', {}).get('title', 'Unknown')
            text = result.get('text', '')
            method = result.get('retrieval_method', 'vector_search')
            context_parts.append(f"[{i}] {title} (via {method})\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _parse_score(self, score_str: str) -> float:
        """Parse score string to float."""
        try:
            # Extract numeric value from string
            import re
            match = re.search(r'(\d*\.?\d+)', str(score_str))
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return 0.5  # Default


class EnhancedRAG(dspy.Module):
    """Enhanced RAG with multi-chain comparison and quality assessment."""
    
    def __init__(self, retriever: AdaptiveRetriever, state_manager: StateManager):
        super().__init__()
        self.retriever = retriever
        self.state_manager = state_manager
        
        # Load optimized RAG components
        self.answerer: Any = (
            state_manager.load_program("enhanced_answerer") or
            dspy.MultiChainComparison(AnswerWithCitations, M=3)  # Generate 3 chains, pick best
        )
        self.synthesizer: Any = (
            state_manager.load_program("concept_synthesizer") or
            dspy.ChainOfThought(ConceptSynthesis)
        )
    
    async def forward(self, question: str, intent: str = "semantic") -> dspy.Prediction:
        """Enhanced RAG with adaptive retrieval and multi-chain reasoning."""
        
        # Adaptive retrieval
        retriever_any: Any = self.retriever
        retrieval_result = retriever_any(query=question, intent=intent)
        if inspect.isawaitable(retrieval_result):
            retrieval_pred = await retrieval_result
        else:
            retrieval_pred = retrieval_result
            
        context: str = str(getattr(retrieval_pred, 'context', ''))
        results: List[Dict[str, Any]] = cast(List[Dict[str, Any]], getattr(retrieval_pred, 'results', []))
        
        try:
            # Multi-chain answer generation
            answerer_any: Any = self.answerer
            answer_result = answerer_any(
                context_chunks=context,
                question=question,
                retrieval_method=intent
            )
            if inspect.isawaitable(answer_result):
                answer_pred = await answer_result
            else:
                answer_pred = answer_result
            
            # If we have multiple results, try synthesis
            if isinstance(results, list) and len(results) > 3:
                synthesizer_any: Any = self.synthesizer
                synthesis_result = synthesizer_any(
                    sources=context,
                    topic=question
                )
                if inspect.isawaitable(synthesis_result):
                    synthesis_pred = await synthesis_result
                else:
                    synthesis_pred = synthesis_result
                
                # Combine answer with synthesis insights
                answer_text = str(getattr(answer_pred, 'answer', ''))
                synthesis_text = str(getattr(synthesis_pred, 'synthesis', ''))
                combined_answer = f"{answer_text}\n\nAdditional Insights: {synthesis_text}"
                
                return dspy.Prediction(
                    question=question,
                    answer=combined_answer,
                    context=context,
                    citations=getattr(answer_pred, 'citations', []),
                    confidence=str(getattr(answer_pred, 'confidence', '0.8')),
                    retrieval_method=intent,
                    synthesis_insights=synthesis_text,
                    knowledge_gaps=getattr(synthesis_pred, 'knowledge_gaps', 'None identified')
                )
            else:
                return dspy.Prediction(
                    question=question,
                    answer=str(getattr(answer_pred, 'answer', '')),
                    context=context,
                    citations=getattr(answer_pred, 'citations', []),
                    confidence=str(getattr(answer_pred, 'confidence', '0.8')),
                    retrieval_method=intent
                )
                
        except Exception as e:
            logger.error(f"Enhanced RAG failed: {e}")
            # Fallback to simple answer
            return dspy.Prediction(
                question=question,
                answer="I encountered an error while processing your question. Please try rephrasing it.",
                context=context,
                citations=[],
                confidence="0.1",
                retrieval_method=intent,
                error=str(e)
            )


class AdaptiveRAGProgram(dspy.Module):
    """Main adaptive RAG program that routes queries and orchestrates retrieval/generation."""
    
    def __init__(self, unified_store: UnifiedStore, state_dir: Path):
        super().__init__()
        self.unified_store = unified_store
        self.state_manager = StateManager(state_dir)
        
        # Initialize components
        self.router: QueryRouter = QueryRouter(self.state_manager)
        self.retriever: AdaptiveRetriever = AdaptiveRetriever(unified_store, self.state_manager)
        self.rag: EnhancedRAG = EnhancedRAG(self.retriever, self.state_manager)
        
        logger.info(f"Initialized AdaptiveRAGProgram with state dir: {state_dir}")
    
    async def forward(self, question: str) -> dspy.Prediction:
        """Main forward pass: route query -> retrieve -> generate answer."""
        
        # Route query to determine intent
        router_any: Any = self.router
        routing_result = router_any(query=question)
        if inspect.isawaitable(routing_result):
            routing_pred = await routing_result
        else:
            routing_pred = routing_result
            
        intent = str(getattr(routing_pred, 'intent', 'semantic'))
        
        # Enhanced RAG with adaptive retrieval
        rag_any: Any = self.rag
        rag_result = rag_any(question=question, intent=intent)
        if inspect.isawaitable(rag_result):
            rag_pred = await rag_result
        else:
            rag_pred = rag_result
        
        # Add routing information
        try:
            rag_pred.query_intent = intent
            rag_pred.intent_confidence = getattr(routing_pred, 'confidence', '0.0')
            rag_pred.intent_reasoning = getattr(routing_pred, 'reasoning', '')
        except Exception:
            # If rag_pred is not a proper Prediction, coerce it
            rag_pred = dspy.Prediction(
                question=question,
                context=str(getattr(rag_pred, 'context', '')),
                answer=str(getattr(rag_pred, 'answer', '')),
                query_intent=intent,
                intent_confidence=str(getattr(routing_pred, 'confidence', '0.0')),
                intent_reasoning=str(getattr(routing_pred, 'reasoning', '')),
            )

        return rag_pred
    
    def save_optimized_state(self) -> None:
        """Save all optimized components."""
        try:
            self.state_manager.save_program(self.router.classifier, "query_intent_classifier")
            self.state_manager.save_program(self.retriever.similarity_judge, "similarity_judge")
            self.state_manager.save_program(self.retriever.context_expander, "context_expander")
            self.state_manager.save_program(self.retriever.quality_assessor, "quality_assessor")
            self.state_manager.save_program(self.rag.answerer, "enhanced_answerer")
            self.state_manager.save_program(self.rag.synthesizer, "concept_synthesizer")
            logger.info("Saved all optimized program states")
        except Exception as e:
            logger.error(f"Failed to save optimized state: {e}")
