#!/usr/bin/env python3
"""
Evaluation metrics for Graph RAG MCP Server.
Provides standardized metrics for measuring performance and quality.
"""

from typing import Dict, List, Any, Optional
import time
from pydantic import BaseModel, Field


class SearchMetrics(BaseModel):
    """Metrics for search operations"""
    query: str
    total_results: int
    response_time_ms: float
    precision_at_k: Optional[float] = Field(default=None)
    recall_at_k: Optional[float] = Field(default=None)
    relevance_scores: List[float] = Field(default_factory=list)
    strategy_used: Optional[str] = Field(default=None)
    
    def avg_relevance(self) -> float:
        """Calculate average relevance score"""
        if not self.relevance_scores:
            return 0.0
        return sum(self.relevance_scores) / len(self.relevance_scores)


class GraphMetrics(BaseModel):
    """Metrics for graph operations"""
    operation: str
    node_count: int
    edge_count: int
    depth_reached: int
    response_time_ms: float
    connectivity_score: Optional[float] = Field(default=None)
    
    def nodes_per_depth(self) -> float:
        """Calculate average nodes discovered per depth level"""
        if self.depth_reached == 0:
            return float(self.node_count)
        return self.node_count / self.depth_reached


class QAMetrics(BaseModel):
    """Metrics for question answering"""
    question: str
    answer_generated: bool
    response_time_ms: float
    context_chunks_used: int
    answer_length: int
    confidence_score: Optional[float] = Field(default=None)
    factual_accuracy: Optional[float] = Field(default=None, description="If ground truth available")
    
    def context_efficiency(self) -> float:
        """Answer length per context chunk used"""
        if self.context_chunks_used == 0:
            return 0.0
        return self.answer_length / self.context_chunks_used


class EvalMetrics:
    """Centralized metrics collection and analysis"""
    
    def __init__(self):
        self.search_metrics: List[SearchMetrics] = []
        self.graph_metrics: List[GraphMetrics] = []
        self.qa_metrics: List[QAMetrics] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start_timing(self) -> None:
        """Start timing the evaluation"""
        self.start_time = time.time()
    
    def end_timing(self) -> None:
        """End timing the evaluation"""
        self.end_time = time.time()
    
    def total_duration_ms(self) -> float:
        """Get total evaluation duration in milliseconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
    
    def record_search(self, query: str, results: List[Any], 
                     response_time_ms: float, strategy_used: Optional[str] = None) -> SearchMetrics:
        """Record search operation metrics"""
        # Calculate basic relevance scores (placeholder - would use actual relevance in real eval)
        relevance_scores = [0.8, 0.7, 0.6, 0.5, 0.4][:len(results)]  # Mock scores
        
        metrics = SearchMetrics(
            query=query,
            total_results=len(results),
            response_time_ms=response_time_ms,
            relevance_scores=relevance_scores,
            strategy_used=strategy_used
        )
        
        self.search_metrics.append(metrics)
        return metrics
    
    def record_graph_operation(self, operation: str, result: Dict[str, Any], 
                              response_time_ms: float) -> GraphMetrics:
        """Record graph operation metrics"""
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])
        
        metrics = GraphMetrics(
            operation=operation,
            node_count=len(nodes),
            edge_count=len(edges),
            depth_reached=1,  # Would calculate actual depth in real implementation
            response_time_ms=response_time_ms
        )
        
        self.graph_metrics.append(metrics)
        return metrics
    
    def record_qa(self, question: str, answer: str, context_used: int,
                 response_time_ms: float) -> QAMetrics:
        """Record question answering metrics"""
        metrics = QAMetrics(
            question=question,
            answer_generated=bool(answer.strip()),
            response_time_ms=response_time_ms,
            context_chunks_used=context_used,
            answer_length=len(answer.split()) if answer else 0
        )
        
        self.qa_metrics.append(metrics)
        return metrics
    
    def search_summary(self) -> Dict[str, Any]:
        """Generate search metrics summary"""
        if not self.search_metrics:
            return {}
        
        return {
            'total_searches': len(self.search_metrics),
            'avg_response_time_ms': sum(m.response_time_ms for m in self.search_metrics) / len(self.search_metrics),
            'avg_results_per_query': sum(m.total_results for m in self.search_metrics) / len(self.search_metrics),
            'avg_relevance': sum(m.avg_relevance() for m in self.search_metrics) / len(self.search_metrics),
            'strategies_used': list(set(m.strategy_used for m in self.search_metrics if m.strategy_used))
        }
    
    def graph_summary(self) -> Dict[str, Any]:
        """Generate graph metrics summary"""
        if not self.graph_metrics:
            return {}
        
        return {
            'total_operations': len(self.graph_metrics),
            'avg_response_time_ms': sum(m.response_time_ms for m in self.graph_metrics) / len(self.graph_metrics),
            'avg_nodes_per_operation': sum(m.node_count for m in self.graph_metrics) / len(self.graph_metrics),
            'avg_edges_per_operation': sum(m.edge_count for m in self.graph_metrics) / len(self.graph_metrics),
            'operations_types': list(set(m.operation for m in self.graph_metrics))
        }
    
    def qa_summary(self) -> Dict[str, Any]:
        """Generate QA metrics summary"""
        if not self.qa_metrics:
            return {}
        
        successful_answers = [m for m in self.qa_metrics if m.answer_generated]
        
        return {
            'total_questions': len(self.qa_metrics),
            'successful_answers': len(successful_answers),
            'success_rate': len(successful_answers) / len(self.qa_metrics),
            'avg_response_time_ms': sum(m.response_time_ms for m in self.qa_metrics) / len(self.qa_metrics),
            'avg_answer_length': sum(m.answer_length for m in successful_answers) / len(successful_answers) if successful_answers else 0,
            'avg_context_efficiency': sum(m.context_efficiency() for m in successful_answers) / len(successful_answers) if successful_answers else 0
        }
    
    def overall_summary(self) -> Dict[str, Any]:
        """Generate overall metrics summary"""
        return {
            'evaluation_duration_ms': self.total_duration_ms(),
            'search_metrics': self.search_summary(),
            'graph_metrics': self.graph_summary(),
            'qa_metrics': self.qa_summary(),
            'total_operations': len(self.search_metrics) + len(self.graph_metrics) + len(self.qa_metrics)
        }
    
    def performance_score(self) -> float:
        """Calculate overall performance score (0-1)"""
        scores = []
        
        # Search performance (based on response time and relevance)
        if self.search_metrics:
            avg_response_time = sum(m.response_time_ms for m in self.search_metrics) / len(self.search_metrics)
            avg_relevance = sum(m.avg_relevance() for m in self.search_metrics) / len(self.search_metrics)
            
            # Score based on response time (faster = better, cap at 1000ms)
            time_score = max(0.0, 1.0 - (avg_response_time / 1000.0))
            
            # Combine time and relevance
            search_score = (time_score * 0.3 + avg_relevance * 0.7)
            scores.append(search_score)
        
        # Graph performance (based on connectivity and response time)
        if self.graph_metrics:
            avg_response_time = sum(m.response_time_ms for m in self.graph_metrics) / len(self.graph_metrics)
            avg_connectivity = sum(m.nodes_per_depth() for m in self.graph_metrics) / len(self.graph_metrics)
            
            time_score = max(0.0, 1.0 - (avg_response_time / 2000.0))  # Graph ops can be slower
            connectivity_score = min(1.0, avg_connectivity / 10)  # Normalize connectivity
            
            graph_score = (time_score * 0.4 + connectivity_score * 0.6)
            scores.append(graph_score)
        
        # QA performance (based on success rate and efficiency)
        if self.qa_metrics:
            qa_summary = self.qa_summary()
            success_rate = qa_summary['success_rate']
            avg_response_time = qa_summary['avg_response_time_ms']
            
            time_score = max(0.0, 1.0 - (avg_response_time / 3000.0))  # QA can be slowest
            
            qa_score = (success_rate * 0.7 + time_score * 0.3)
            scores.append(qa_score)
        
        # Return average of all component scores
        return sum(scores) / len(scores) if scores else 0.0


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000