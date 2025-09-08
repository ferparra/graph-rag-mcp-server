"""
DSPy Optimizer with Persistent State for Graph RAG System

This module implements MIPROv2 optimization with persistent state management,
allowing the system to continuously improve across ephemeral uvx connections.
"""

from __future__ import annotations
import dspy
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import json
import time
import threading
import logging
from datetime import datetime

# Support both package and module execution contexts
try:
    from config import settings
    from dspy_eval import VaultEvaluator, CustomMetrics
    from dspy_programs import AdaptiveRAGProgram, AdaptiveRetriever, EnhancedRAG
    from unified_store import UnifiedStore
except ImportError:  # When imported as part of a package
    from .config import settings
    from .dspy_eval import VaultEvaluator, CustomMetrics
    from .dspy_programs import AdaptiveRAGProgram, AdaptiveRetriever, EnhancedRAG
    from .unified_store import UnifiedStore

logger = logging.getLogger(__name__)


class OptimizationScheduler:
    """Manages scheduled optimization runs with persistent state."""
    
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.schedule_file = state_dir / "optimization_schedule.json"
        self.lock_file = state_dir / "optimization.lock"
        
    def should_run_optimization(self, component_name: str = "main") -> bool:
        """Check if optimization should run based on schedule."""
        try:
            # Check if optimization is already running
            if self.lock_file.exists():
                # Check if lock is stale (>2 hours)
                lock_age = time.time() - self.lock_file.stat().st_mtime
                if lock_age < 7200:  # 2 hours
                    return False
                else:
                    # Remove stale lock
                    self.lock_file.unlink()
            
            # Load schedule
            schedule = self._load_schedule()
            last_run = schedule.get(component_name, 0)
            
            # Check if enough time has passed
            hours_since_last = (time.time() - last_run) / 3600
            return hours_since_last >= settings.dspy_optimization_interval_hours
            
        except Exception as e:
            logger.error(f"Error checking optimization schedule: {e}")
            return False
    
    def mark_optimization_start(self, component_name: str = "main") -> None:
        """Mark optimization as started."""
        try:
            # Create lock file
            self.lock_file.touch()
            
            # Update schedule
            schedule = self._load_schedule()
            schedule[f"{component_name}_started"] = time.time()
            self._save_schedule(schedule)
            
        except Exception as e:
            logger.error(f"Error marking optimization start: {e}")
    
    def mark_optimization_complete(self, component_name: str = "main", success: bool = True) -> None:
        """Mark optimization as complete."""
        try:
            # Remove lock file
            if self.lock_file.exists():
                self.lock_file.unlink()
            
            # Update schedule
            schedule = self._load_schedule()
            if success:
                schedule[component_name] = time.time()
            schedule.pop(f"{component_name}_started", None)
            self._save_schedule(schedule)
            
        except Exception as e:
            logger.error(f"Error marking optimization complete: {e}")
    
    def _load_schedule(self) -> Dict[str, float]:
        """Load optimization schedule."""
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_schedule(self, schedule: Dict[str, float]) -> None:
        """Save optimization schedule."""
        try:
            with open(self.schedule_file, 'w') as f:
                json.dump(schedule, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving schedule: {e}")


class ProgramOptimizer:
    """Main optimizer for DSPy programs with persistent state management."""
    
    def __init__(self, unified_store: UnifiedStore, state_dir: Path):
        self.unified_store = unified_store
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = VaultEvaluator(unified_store, state_dir)
        self.scheduler = OptimizationScheduler(state_dir)
        self.metrics = CustomMetrics()
        
        # Optimization history
        self.history_file = state_dir / "optimization_history.json"
        
        logger.info(f"Initialized ProgramOptimizer with state dir: {state_dir}")
    
    def optimize_adaptive_rag(self, program: AdaptiveRAGProgram, 
                             force: bool = False) -> Optional[AdaptiveRAGProgram]:
        """Optimize the full adaptive RAG program."""
        
        if not force and not self.scheduler.should_run_optimization("adaptive_rag"):
            logger.info("Skipping optimization - not due yet")
            return None
        
        logger.info("Starting adaptive RAG optimization...")
        self.scheduler.mark_optimization_start("adaptive_rag")
        
        try:
            # Create evaluation dataset
            dataset = self.evaluator.create_evaluation_dataset(
                max_examples=settings.dspy_max_examples
            )
            
            if not dataset:
                logger.warning("No evaluation dataset available for optimization")
                return None
            
            # Split dataset
            train_size = min(len(dataset), 30)
            trainset = dataset[:train_size]
            
            # Optimize individual components
            optimized_components = {}
            
            # 1. Optimize query router
            router_optimized = self._optimize_component(
                program.router.classifier,
                trainset,
                self.metrics.intent_classification_accuracy,
                "query_router"
            )
            if router_optimized:
                optimized_components["router"] = router_optimized
                program.router.classifier = router_optimized
            
            # 2. Optimize retriever components
            retriever_components = self._optimize_retriever_components(
                program.retriever, trainset
            )
            optimized_components.update(retriever_components)
            
            # 3. Optimize RAG components
            rag_components = self._optimize_rag_components(
                program.rag, trainset
            )
            optimized_components.update(rag_components)
            
            # Evaluate final optimized program
            final_score = self.evaluator.quick_eval(program, num_samples=10)
            
            # Log optimization results
            self._log_optimization_results("adaptive_rag", {
                "components_optimized": list(optimized_components.keys()),
                "final_score": final_score,
                "dataset_size": len(dataset),
                "train_size": train_size
            })
            
            # Save optimized program state
            program.save_optimized_state()
            
            self.scheduler.mark_optimization_complete("adaptive_rag", success=True)
            logger.info(f"Adaptive RAG optimization complete. Final score: {final_score:.3f}")
            
            return program
            
        except Exception as e:
            logger.error(f"Adaptive RAG optimization failed: {e}")
            self.scheduler.mark_optimization_complete("adaptive_rag", success=False)
            return None
    
    def _optimize_component(self, component: dspy.Module, dataset: List[dspy.Example],
                           metric: Callable, component_name: str) -> Optional[dspy.Module]:
        """Optimize a single DSPy component."""
        try:
            from dspy.teleprompt import MIPROv2
            
            logger.info(f"Optimizing component: {component_name}")
            
            # Initialize optimizer
            optimizer = MIPROv2(
                metric=metric,
                auto=settings.dspy_auto_mode,
                num_threads=2  # Conservative for stability
            )
            
            # Run optimization
            optimized = optimizer.compile(
                component,
                trainset=dataset,
                max_bootstrapped_demos=settings.dspy_bootstrap_demos,
                max_labeled_demos=settings.dspy_labeled_demos
            )
            
            # Get optimization metrics
            try:
                opt_metrics = {
                    "component": component_name,
                    "improvement": "optimized",
                    "auto_mode": settings.dspy_auto_mode
                }
            except Exception:
                opt_metrics = {"component": component_name}
            
            self._log_component_optimization(component_name, opt_metrics)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Failed to optimize component {component_name}: {e}")
            return None
    
    def _optimize_retriever_components(self, retriever: AdaptiveRetriever, 
                                     dataset: List[dspy.Example]) -> Dict[str, dspy.Module]:
        """Optimize retriever components."""
        optimized = {}
        
        # Optimize similarity judge
        similarity_opt = self._optimize_component(
            retriever.similarity_judge,
            dataset,
            self.metrics.retrieval_relevance,
            "similarity_judge"
        )
        if similarity_opt:
            optimized["similarity_judge"] = similarity_opt
            retriever.similarity_judge = similarity_opt
        
        # Optimize context expander
        context_opt = self._optimize_component(
            retriever.context_expander,
            dataset,
            self.metrics.semantic_f1_with_context,
            "context_expander"
        )
        if context_opt:
            optimized["context_expander"] = context_opt
            retriever.context_expander = context_opt
        
        # Optimize quality assessor
        quality_opt = self._optimize_component(
            retriever.quality_assessor,
            dataset,
            self.metrics.comprehensive_rag_score,
            "quality_assessor"
        )
        if quality_opt:
            optimized["quality_assessor"] = quality_opt
            retriever.quality_assessor = quality_opt
        
        return optimized
    
    def _optimize_rag_components(self, rag: EnhancedRAG, 
                               dataset: List[dspy.Example]) -> Dict[str, dspy.Module]:
        """Optimize RAG components."""
        optimized = {}
        
        # Optimize main answerer
        answerer_opt = self._optimize_component(
            rag.answerer,
            dataset,
            self.metrics.comprehensive_rag_score,
            "enhanced_answerer"
        )
        if answerer_opt:
            optimized["enhanced_answerer"] = answerer_opt
            rag.answerer = answerer_opt
        
        # Optimize synthesizer
        synthesizer_opt = self._optimize_component(
            rag.synthesizer,
            dataset,
            self.metrics.semantic_f1_with_context,
            "concept_synthesizer"
        )
        if synthesizer_opt:
            optimized["concept_synthesizer"] = synthesizer_opt
            rag.synthesizer = synthesizer_opt
        
        return optimized
    
    def run_background_optimization(self, program: AdaptiveRAGProgram) -> None:
        """Run optimization in background thread."""
        def optimize():
            try:
                self.optimize_adaptive_rag(program)
            except Exception as e:
                logger.error(f"Background optimization failed: {e}")
        
        thread = threading.Thread(target=optimize, daemon=True)
        thread.start()
        logger.info("Started background optimization thread")
    
    def _log_optimization_results(self, program_name: str, results: Dict[str, Any]) -> None:
        """Log optimization results to history."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "program": program_name,
                "results": results
            }
            
            # Load existing history
            history = []
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new entry
            history.append(entry)
            
            # Keep only last 50 entries
            history = history[-50:]
            
            # Save updated history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log optimization results: {e}")
    
    def _log_component_optimization(self, component_name: str, metrics: Dict[str, Any]) -> None:
        """Log individual component optimization."""
        try:
            component_log_file = self.state_dir / f"{component_name}_optimization.json"
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            
            # Load existing log
            log_data = []
            if component_log_file.exists():
                with open(component_log_file, 'r') as f:
                    log_data = json.load(f)
            
            # Add new entry
            log_data.append(entry)
            
            # Keep only last 20 entries
            log_data = log_data[-20:]
            
            # Save updated log
            with open(component_log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log component optimization: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        try:
            status = {
                "optimization_enabled": settings.dspy_optimize_enabled,
                "state_dir": str(self.state_dir),
                "next_optimization_due": False,
                "optimization_history": []
            }
            
            # Check if optimization is due
            status["next_optimization_due"] = self.scheduler.should_run_optimization("adaptive_rag")
            
            # Load recent history
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    status["optimization_history"] = history[-5:]  # Last 5 entries
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            return {"error": str(e)}
    
    def force_optimization(self, program: AdaptiveRAGProgram) -> Dict[str, Any]:
        """Force immediate optimization."""
        logger.info("Forcing immediate optimization...")
        
        result = self.optimize_adaptive_rag(program, force=True)
        
        if result:
            return {
                "success": True,
                "message": "Optimization completed successfully",
                "program_optimized": True
            }
        else:
            return {
                "success": False,
                "message": "Optimization failed or was skipped",
                "program_optimized": False
            }


class OptimizationManager:
    """High-level manager for all optimization operations."""
    
    def __init__(self, unified_store: UnifiedStore):
        self.unified_store = unified_store
        self.state_dir = settings.dspy_state_dir
        self.optimizer = ProgramOptimizer(unified_store, self.state_dir)
        self._current_program: Optional[AdaptiveRAGProgram] = None
    
    def initialize_or_load_program(self) -> AdaptiveRAGProgram:
        """Initialize new program or load optimized one from state."""
        if self._current_program is None:
            self._current_program = AdaptiveRAGProgram(self.unified_store, self.state_dir)
            
            # If optimization is enabled, check for background optimization
            if settings.dspy_optimize_enabled:
                if self.optimizer.scheduler.should_run_optimization("adaptive_rag"):
                    logger.info("Scheduling background optimization...")
                    self.optimizer.run_background_optimization(self._current_program)
        
        return self._current_program
    
    def get_program(self) -> AdaptiveRAGProgram:
        """Get current optimized program."""
        return self.initialize_or_load_program()
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization."""
        program = self.get_program()
        return self.optimizer.force_optimization(program)
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimization status."""
        return self.optimizer.get_optimization_status()