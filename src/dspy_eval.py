"""
DSPy Evaluation Framework for Graph RAG System

This module provides automatic evaluation dataset generation and custom metrics
for optimizing DSPy programs based on vault content and usage patterns.
"""

from __future__ import annotations
import dspy
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import random
import re
import logging
import inspect
from dataclasses import dataclass

# Support both package and module execution contexts
try:
    from unified_store import UnifiedStore
    from config import settings
except ImportError:  # When imported as part of a package
    from .unified_store import UnifiedStore
    from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """Evaluation example with question, expected intent, and ideal answer."""
    question: str
    expected_intent: str
    ideal_answer: Optional[str] = None
    relevant_chunks: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class VaultDatasetGenerator:
    """Automatically generates evaluation datasets from vault content patterns."""
    
    def __init__(self, unified_store: UnifiedStore, cache_path: Optional[Path] = None):
        self.unified_store = unified_store
        self.cache_path = cache_path or (settings.dspy_state_dir / "eval_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    def generate_dataset(self, max_examples: int = 50, refresh_cache: bool = False) -> List[dspy.Example]:
        """Generate evaluation dataset from vault patterns."""
        
        # Try to load from cache first
        if not refresh_cache and self.cache_path.exists():
            try:
                cached_examples = self._load_cached_dataset()
                if len(cached_examples) >= max_examples // 2:
                    logger.info(f"Loaded {len(cached_examples)} examples from cache")
                    return cached_examples[:max_examples]
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}")
        
        # Generate new dataset
        examples = []
        
        # Generate different types of evaluation examples
        examples.extend(self._generate_header_questions(max_examples // 5))
        examples.extend(self._generate_link_relationship_questions(max_examples // 5))
        examples.extend(self._generate_tag_categorical_questions(max_examples // 5))
        examples.extend(self._generate_concept_questions(max_examples // 5))
        examples.extend(self._generate_factual_questions(max_examples // 5))
        
        # Shuffle and limit
        random.shuffle(examples)
        examples = examples[:max_examples]
        
        # Convert to DSPy examples
        dspy_examples = []
        for ex in examples:
            try:
                dspy_ex = dspy.Example(
                    question=ex.question,
                    expected_intent=ex.expected_intent,
                    ideal_answer=ex.ideal_answer or "Generated example",
                    relevant_chunks=ex.relevant_chunks or []
                ).with_inputs("question")
                dspy_examples.append(dspy_ex)
            except Exception as e:
                logger.error(f"Failed to create DSPy example: {e}")
        
        # Cache the results
        try:
            self._cache_dataset(examples)
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")
        
        logger.info(f"Generated {len(dspy_examples)} evaluation examples")
        return dspy_examples
    
    def _generate_header_questions(self, max_count: int) -> List[EvalExample]:
        """Generate questions based on document headers."""
        examples = []
        
        try:
            # Get sample of chunks with headers
            all_chunks = self.unified_store._collection().get(
                where={"header_text": {"$ne": ""}},
                include=['metadatas', 'documents'],
                limit=max_count * 3
            )
            
            metadatas = all_chunks.get('metadatas', [])
            documents = all_chunks.get('documents', [])
            
            # Ensure metadatas and documents are lists
            if not isinstance(metadatas, list):
                metadatas = []
            if not isinstance(documents, list):
                documents = []
            
            for i, meta in enumerate(metadatas[:max_count]):
                if i >= len(documents) or not isinstance(meta, dict):
                    break
                    
                header_text = meta.get('header_text', '')
                if not header_text or not isinstance(header_text, str):
                    continue
                
                # Generate question from header
                questions = [
                    f"What is {header_text.lower()}?",
                    f"Tell me about {header_text.lower()}",
                    f"Explain {header_text.lower()}",
                    f"What does the section on {header_text.lower()} cover?"
                ]
                
                question = random.choice(questions)
                
                examples.append(EvalExample(
                    question=question,
                    expected_intent="semantic",
                    ideal_answer=f"Information about {header_text}",
                    relevant_chunks=[documents[i][:200]],
                    metadata={"source": "header", "header": header_text}
                ))
                
        except Exception as e:
            logger.error(f"Failed to generate header questions: {e}")
        
        return examples
    
    def _generate_link_relationship_questions(self, max_count: int) -> List[EvalExample]:
        """Generate questions about links and relationships."""
        examples = []
        
        try:
            # Get chunks with links
            all_chunks = self.unified_store._collection().get(
                where={"contains_links": {"$ne": ""}},
                include=['metadatas', 'documents'],
                limit=max_count * 2
            )
            
            metadatas = all_chunks.get('metadatas', [])
            
            # Ensure metadatas is a list
            if not isinstance(metadatas, list):
                metadatas = []
            
            for meta in metadatas[:max_count]:
                if not isinstance(meta, dict):
                    continue
                    
                contains_links = meta.get('contains_links', '')
                if not isinstance(contains_links, str):
                    continue
                    
                links = contains_links.split(',')
                if not links or not links[0]:
                    continue
                
                note_title = meta.get('title', 'Unknown')
                linked_note = links[0].strip()
                
                questions = [
                    f"How is {note_title} related to {linked_note}?",
                    f"What is the connection between {note_title} and {linked_note}?",
                    f"Show me the relationship between {note_title} and {linked_note}",
                    f"What links {note_title} to {linked_note}?"
                ]
                
                question = random.choice(questions)
                
                examples.append(EvalExample(
                    question=question,
                    expected_intent="graph",
                    ideal_answer=f"Relationship between {note_title} and {linked_note}",
                    metadata={"source": "link", "notes": [note_title, linked_note]}
                ))
                
        except Exception as e:
            logger.error(f"Failed to generate link relationship questions: {e}")
        
        return examples
    
    def _generate_tag_categorical_questions(self, max_count: int) -> List[EvalExample]:
        """Generate questions about tags and categories."""
        examples = []
        
        try:
            # Get chunks with tags
            all_chunks = self.unified_store._collection().get(
                where={"tags": {"$ne": ""}},
                include=['metadatas'],
                limit=max_count * 2
            )
            
            metadatas = all_chunks.get('metadatas', [])
            
            # Ensure metadatas is a list
            if not isinstance(metadatas, list):
                metadatas = []
            
            # Collect unique tags
            all_tags = set()
            for meta in metadatas:
                if not isinstance(meta, dict):
                    continue
                    
                tags_str = meta.get('tags', '')
                if not isinstance(tags_str, str):
                    continue
                    
                tags = tags_str.split(',')
                all_tags.update([tag.strip() for tag in tags if tag.strip()])
            
            # Generate tag-based questions
            for tag in list(all_tags)[:max_count]:
                questions = [
                    f"What notes are tagged with #{tag}?",
                    f"Show me everything tagged #{tag}",
                    f"Find notes with tag {tag}",
                    f"What content is categorized as {tag}?"
                ]
                
                question = random.choice(questions)
                
                examples.append(EvalExample(
                    question=question,
                    expected_intent="categorical",
                    ideal_answer=f"Notes and content tagged with {tag}",
                    metadata={"source": "tag", "tag": tag}
                ))
                
        except Exception as e:
            logger.error(f"Failed to generate tag questions: {e}")
        
        return examples
    
    def _generate_concept_questions(self, max_count: int) -> List[EvalExample]:
        """Generate questions about concepts and definitions."""
        examples = []
        
        try:
            # Sample random documents
            all_chunks = self.unified_store._collection().get(
                include=['metadatas', 'documents'],
                limit=max_count * 3
            )
            
            documents = all_chunks.get('documents', [])
            metadatas = all_chunks.get('metadatas', [])
            
            # Ensure both are lists
            if not isinstance(documents, list):
                documents = []
            if not isinstance(metadatas, list):
                metadatas = []
            
            for i, doc in enumerate(documents[:max_count]):
                if i >= len(metadatas):
                    break
                
                # Extract potential concepts (capitalized words, technical terms)
                concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', doc)
                concepts = [c for c in concepts if len(c) > 3 and c not in ['The', 'This', 'That', 'With', 'From']]
                
                if concepts:
                    concept = random.choice(concepts)
                    
                    questions = [
                        f"What is {concept}?",
                        f"Define {concept}",
                        f"Explain the concept of {concept}",
                        f"Tell me about {concept}"
                    ]
                    
                    question = random.choice(questions)
                    
                    examples.append(EvalExample(
                        question=question,
                        expected_intent="semantic",
                        ideal_answer=f"Definition and explanation of {concept}",
                        relevant_chunks=[doc[:300]],
                        metadata={"source": "concept", "concept": concept}
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to generate concept questions: {e}")
        
        return examples
    
    def _generate_factual_questions(self, max_count: int) -> List[EvalExample]:
        """Generate specific factual questions."""
        examples = []
        
        try:
            # Sample documents with specific information
            all_chunks = self.unified_store._collection().get(
                include=['metadatas', 'documents'],
                limit=max_count * 2
            )
            
            documents = all_chunks.get('documents', [])
            metadatas = all_chunks.get('metadatas', [])
            
            # Ensure both are lists
            if not isinstance(documents, list):
                documents = []
            if not isinstance(metadatas, list):
                metadatas = []
            
            for i, doc in enumerate(documents[:max_count]):
                if i >= len(metadatas) or not isinstance(metadatas[i], dict):
                    break
                
                note_title = metadatas[i].get('title', 'Unknown')
                
                # Generate specific questions about the note
                questions = [
                    f"What specific information is in {note_title}?",
                    f"What are the key points from {note_title}?",
                    f"Summarize the main ideas in {note_title}",
                    f"What details does {note_title} provide?"
                ]
                
                question = random.choice(questions)
                
                examples.append(EvalExample(
                    question=question,
                    expected_intent="specific",
                    ideal_answer=f"Specific information from {note_title}",
                    relevant_chunks=[doc[:400]],
                    metadata={"source": "factual", "note": note_title}
                ))
                
        except Exception as e:
            logger.error(f"Failed to generate factual questions: {e}")
        
        return examples
    
    def _cache_dataset(self, examples: List[EvalExample]) -> None:
        """Cache generated dataset."""
        cache_data = []
        for ex in examples:
            cache_data.append({
                "question": ex.question,
                "expected_intent": ex.expected_intent,
                "ideal_answer": ex.ideal_answer,
                "relevant_chunks": ex.relevant_chunks,
                "metadata": ex.metadata
            })
        
        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _load_cached_dataset(self) -> List[dspy.Example]:
        """Load cached dataset."""
        with open(self.cache_path, 'r') as f:
            cache_data = json.load(f)
        
        examples = []
        for data in cache_data:
            ex = dspy.Example(
                question=data["question"],
                expected_intent=data["expected_intent"],
                ideal_answer=data.get("ideal_answer", ""),
                relevant_chunks=data.get("relevant_chunks", [])
            ).with_inputs("question")
            examples.append(ex)
        
        return examples


class CustomMetrics:
    """Custom evaluation metrics for Graph RAG system."""
    
    @staticmethod
    def semantic_f1_with_context(prediction, example) -> float:
        """Semantic F1 that considers context quality."""
        try:
            # Try to import SemanticF1 dynamically, fall back if not available
            try:
                import importlib
                metrics_mod = importlib.import_module('dspy.evaluate.metrics')
                semantic_cls = getattr(metrics_mod, 'SemanticF1', None)
                if semantic_cls is not None:
                    semantic_f1 = semantic_cls()
                    base_score = semantic_f1(prediction, example)
                else:
                    raise ImportError('SemanticF1 not available')
            except Exception:
                # Use a simple word overlap metric as fallback
                pred_words = set(str(prediction).lower().split())
                example_words = set(str(example).lower().split())
                intersection = pred_words & example_words
                if not (pred_words or example_words):
                    base_score = 0.0
                else:
                    precision = len(intersection) / len(pred_words) if pred_words else 0.0
                    recall = len(intersection) / len(example_words) if example_words else 0.0
                    base_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Context quality bonus
            context_bonus = 0.0
            if hasattr(prediction, 'context') and prediction.context:
                # Check if context contains relevant information
                context_lower = prediction.context.lower()
                question_lower = example.question.lower()
                
                # Simple keyword overlap
                question_words = set(question_lower.split())
                context_words = set(context_lower.split())
                overlap = len(question_words & context_words) / len(question_words)
                context_bonus = min(overlap * 0.2, 0.2)  # Max 20% bonus
            
            return min(base_score + context_bonus, 1.0)
            
        except Exception as e:
            logger.error(f"Semantic F1 evaluation failed: {e}")
            return 0.0
    
    @staticmethod
    def intent_classification_accuracy(prediction, example) -> float:
        """Accuracy for intent classification."""
        try:
            predicted_intent = getattr(prediction, 'intent', '').lower()
            expected_intent = getattr(example, 'expected_intent', '').lower()
            return 1.0 if predicted_intent == expected_intent else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def citation_quality(prediction, example) -> float:
        """Quality of citations provided."""
        try:
            citations = getattr(prediction, 'citations', [])
            if not citations:
                return 0.0
            
            # Check if citations are well-formed
            valid_citations = 0
            for citation in citations:
                if isinstance(citation, (list, tuple)) and len(citation) >= 2:
                    valid_citations += 1
                elif isinstance(citation, str) and len(citation) > 10:
                    valid_citations += 1
            
            return valid_citations / len(citations) if citations else 0.0
            
        except Exception:
            return 0.0
    
    @staticmethod
    def retrieval_relevance(prediction, example) -> float:
        """Relevance of retrieved context."""
        try:
            context = getattr(prediction, 'context', '')
            question = getattr(example, 'question', '')
            
            if not context or not question:
                return 0.0
            
            # Simple relevance scoring
            question_words = set(question.lower().split())
            context_words = set(context.lower().split())
            
            # Jaccard similarity
            intersection = len(question_words & context_words)
            union = len(question_words | context_words)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    @staticmethod
    def comprehensive_rag_score(prediction, example) -> float:
        """Comprehensive RAG evaluation combining multiple factors."""
        weights = {
            'semantic': 0.4,
            'citation': 0.2,
            'relevance': 0.2,
            'completeness': 0.2
        }
        
        scores = {
            'semantic': CustomMetrics.semantic_f1_with_context(prediction, example),
            'citation': CustomMetrics.citation_quality(prediction, example),
            'relevance': CustomMetrics.retrieval_relevance(prediction, example),
            'completeness': CustomMetrics._answer_completeness(prediction, example)
        }
        
        return sum(scores[key] * weights[key] for key in weights)
    
    @staticmethod
    def _answer_completeness(prediction, example) -> float:
        """Assess answer completeness."""
        try:
            answer = getattr(prediction, 'answer', '')
            if not answer:
                return 0.0
            
            # Simple heuristics for completeness
            word_count = len(answer.split())
            has_structure = any(marker in answer for marker in ['.', ':', '-', '1.', '*'])
            
            # Score based on length and structure
            length_score = min(word_count / 50, 1.0)  # Normalize around 50 words
            structure_score = 0.2 if has_structure else 0.0
            
            return min(length_score + structure_score, 1.0)
            
        except Exception:
            return 0.0


class VaultEvaluator:
    """Main evaluator for DSPy programs using vault-specific metrics."""
    
    def __init__(self, unified_store: UnifiedStore, state_dir: Path):
        self.unified_store = unified_store
        self.state_dir = state_dir
        self.dataset_generator = VaultDatasetGenerator(unified_store, state_dir / "eval_cache.json")
        self.metrics = CustomMetrics()
    
    def create_evaluation_dataset(self, max_examples: int = 50, refresh: bool = False) -> List[dspy.Example]:
        """Create evaluation dataset for optimization."""
        return self.dataset_generator.generate_dataset(max_examples, refresh)
    
    async def evaluate_program(self, program: dspy.Module, dataset: Optional[List[dspy.Example]] = None) -> Dict[str, float]:
        """Evaluate a DSPy program on the vault dataset."""
        
        if dataset is None:
            dataset = self.create_evaluation_dataset(20)  # Smaller for quick eval
        
        if not dataset:
            logger.warning("No evaluation dataset available")
            return {}
        
        # Run evaluation
        try:
            from dspy.evaluate import Evaluate
            
            evaluator = Evaluate(
                devset=dataset,
                metrics=[
                    self.metrics.semantic_f1_with_context,
                    self.metrics.citation_quality,
                    self.metrics.retrieval_relevance,
                    self.metrics.comprehensive_rag_score
                ],
                provide_traceback=True
            )
            
            result = evaluator(program)
            
            # Handle potential async result
            if inspect.isawaitable(result):
                results = await result
            else:
                results = result
            
            # Ensure results is a dict with float values
            if not isinstance(results, dict):
                results = {"evaluation_score": 0.5}
            
            # Convert string values to floats where possible
            float_results = {}
            for key, value in results.items():
                try:
                    float_results[key] = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    float_results[key] = 0.0
            
            logger.info(f"Evaluation results: {float_results}")
            return float_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": 0.0}  # Return 0.0 instead of string for type consistency
    
    async def quick_eval(self, program: dspy.Module, num_samples: int = 5) -> float:
        """Quick evaluation with small sample."""
        dataset = self.create_evaluation_dataset(num_samples)
        results = await self.evaluate_program(program, dataset)
        
        # Return comprehensive score or fallback
        return results.get('comprehensive_rag_score', results.get('semantic_f1_with_context', 0.5))
