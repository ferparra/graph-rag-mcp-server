# DSPy Enhancements for Graph RAG MCP Server

## Overview

The Graph RAG MCP Server now includes sophisticated DSPy enhancements that transform the basic RAG system into a self-improving, multi-strategy intelligent assistant. These enhancements leverage DSPy's most powerful features while maintaining persistent state across ephemeral uvx connections.

## Key Features

### 🎯 **Adaptive Query Routing**
- Intelligent query intent classification (semantic, graph, categorical, specific, analytical)
- Automatic routing to optimal retrieval strategies
- Multi-chain reasoning with confidence scoring

### 🔄 **Self-Improving System**
- Automatic prompt optimization with MIPROv2
- Continuous learning from vault content patterns
- Weekly optimization with persistent state

### 🕸️ **Multi-Hop Graph Reasoning**
- ReAct agent for complex relationship queries
- Tool-based graph traversal with backtracking
- Multi-step reasoning with thought/action/observation loops

### 💾 **Persistent State Management**
- Optimized programs survive uvx ephemeral connections
- State directory configuration for uvx deployments
- Automatic caching and loading of improvements

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query Intent  │    │   Adaptive      │    │   Enhanced      │
│  Classification │───▶│   Retrieval     │───▶│     RAG         │
│   (DSPy Module) │    │  (Multi-Strategy)│    │ (MultiChain)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MIPROv2        │    │   Graph Agent   │    │   State         │
│  Optimizer      │    │   (ReAct)       │    │   Manager       │
│ (Auto-Improve)  │    │  (Multi-hop)    │    │ (Persistence)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration

### Environment Variables for UVX

When using with Claude Desktop via uvx, add these environment variables:

```json
{
  "mcpServers": {
    "graph-rag-obsidian": {
      "command": "uvx",
      "args": ["--python", "3.13", "--from", ".", "graph-rag-mcp-stdio"],
      "cwd": "/path/to/graph-rag-mcp-server",
      "env": {
        "GEMINI_API_KEY": "your-gemini-key",
        "OBSIDIAN_RAG_CHROMA_DIR": "/Users/you/.obsidian-rag/chroma_db",
        "OBSIDIAN_RAG_DSPY_STATE_DIR": "/Users/you/.obsidian-rag/dspy_state",
        "OBSIDIAN_RAG_DSPY_OPTIMIZE_ENABLED": "true",
        "OBSIDIAN_RAG_DSPY_AUTO_MODE": "light"
      }
    }
  }
}
```

### Configuration Options

- `OBSIDIAN_RAG_DSPY_STATE_DIR`: Persistent directory for optimized programs
- `OBSIDIAN_RAG_DSPY_OPTIMIZE_ENABLED`: Enable/disable optimization (default: true)
- `OBSIDIAN_RAG_DSPY_AUTO_MODE`: MIPROv2 mode - light/medium/heavy (default: light)
- `OBSIDIAN_RAG_DSPY_OPTIMIZATION_INTERVAL_HOURS`: Hours between optimizations (default: 168)
- `OBSIDIAN_RAG_DSPY_MAX_EXAMPLES`: Max examples for optimization (default: 50)

## Enhanced Components

### 1. Query Intent Classification

**Signatures:**
- `QueryIntentClassifier`: Analyzes query intent and confidence
- `ContextExpansion`: Determines if additional context is needed
- `ContentQualityAssessment`: Evaluates retrieval completeness

**Intent Types:**
- **Semantic**: General similarity-based queries
- **Graph**: Relationship and connection queries  
- **Categorical**: Tag-based organizational queries
- **Specific**: Targeted fact extraction
- **Analytical**: Complex multi-step reasoning

### 2. Adaptive Retrieval

**Multi-Strategy Approach:**
```python
# Semantic: Vector similarity
results = unified_store.query(query, k=6)

# Graph: Relationship traversal
neighbors = unified_store.get_chunk_neighbors(chunk_id)

# Categorical: Tag-based filtering
results = unified_store.query(query, where={"tags": {"$in": tags}})

# Hybrid: Intelligent expansion
expanded = retriever.expand_context(strategy, k)
```

**Quality Assessment:**
- Completeness scoring (0.0-1.0)
- Diversity filtering to avoid redundancy
- Automatic context expansion when needed

### 3. Enhanced RAG Pipeline

**Multi-Chain Comparison:**
- Generates 3 reasoning chains
- Selects best answer with voting
- Provides confidence scoring

**Answer Synthesis:**
- Combines multiple sources intelligently
- Identifies knowledge gaps
- Structured citations with source tracking

### 4. ReAct Agent for Complex Queries

**Available Tools:**
- `search_semantic`: Vector similarity search
- `traverse_graph`: Multi-hop graph navigation
- `extract_properties`: Targeted fact extraction
- `expand_context`: Dynamic context enrichment
- `find_connections`: Relationship discovery

**Complex Query Patterns:**
- "How does X relate to Y?"
- "Trace the development of concept Z"
- "Compare and analyze A vs B"
- "What's the connection between multiple entities?"

### 5. Automatic Optimization

**MIPROv2 Optimization:**
- Weekly automatic optimization runs
- Optimizes prompts for vault-specific content
- Learns from query patterns and user interactions

**Evaluation Dataset Generation:**
- Header-based questions from document structure
- Link relationship queries from graph connections
- Tag categorical searches from organizational patterns
- Concept definitions from content analysis
- Factual questions from note-specific content

**Custom Metrics:**
- `semantic_f1_with_context`: Context-aware semantic similarity
- `citation_quality`: Quality of source citations
- `retrieval_relevance`: Relevance of retrieved content
- `comprehensive_rag_score`: Overall system performance

## Usage Examples

### Basic Enhanced RAG

```python
from src.dspy_rag import EnhancedVaultSearcher

searcher = EnhancedVaultSearcher(unified_store)

# Automatic intent classification and routing
result = searcher.ask("How does project management relate to team productivity?")

print(f"Answer: {result['answer']}")
print(f"Method: {result['method']}")  # enhanced_adaptive_rag
print(f"Intent: {result['query_intent']}")  # graph
print(f"Citations: {result['citations']}")
```

### Complex Multi-hop Queries

```python
# The system automatically detects complex queries and uses ReAct agent
result = searcher.ask("""
Compare the relationship between agile methodology and team velocity 
across different project types in my vault.
""")

print(f"Method: {result['method']}")  # complex_agent
print(f"Reasoning: {result.get('reasoning_trace', 'N/A')}")
```

### Manual Optimization

```python
# Force immediate optimization
optimization_result = searcher.force_optimization()
print(f"Optimization success: {optimization_result['success']}")

# Check optimization status
status = searcher.get_optimization_status()
print(f"Next optimization due: {status['next_optimization_due']}")
```

## MCP Tools

### New DSPy-Specific Tools

1. **`force_dspy_optimization`**: Trigger immediate optimization
2. **`get_dspy_optimization_status`**: Check optimization metrics and status

### Enhanced Existing Tools

- **`answer_question`**: Now uses adaptive RAG with intent routing
- **`search_notes`**: Enhanced with multi-strategy retrieval
- **`graph_neighbors`**: Improved with ReAct agent capabilities

## Performance Improvements

### Expected Improvements

- **30-50% better answer quality** through optimized prompts
- **Multi-hop reasoning** for complex relationship queries
- **Reduced hallucination** through grounded citations
- **Adaptive context** expansion for incomplete information
- **Self-improving accuracy** over time

### Benchmarking

The system includes built-in evaluation metrics:

```python
# Comprehensive evaluation
score = evaluator.evaluate_program(program, dataset)

# Quick evaluation with small sample
quick_score = evaluator.quick_eval(program, num_samples=5)
```

## State Management

### Persistent Storage Structure

```
.dspy_state/
├── query_intent_classifier.json     # Optimized intent classifier
├── similarity_judge.json            # Optimized retrieval scoring
├── context_expander.json           # Optimized context expansion
├── quality_assessor.json           # Optimized quality assessment
├── enhanced_answerer.json          # Optimized answer generation
├── concept_synthesizer.json        # Optimized synthesis
├── graph_reasoning_agent.json      # Optimized ReAct agent
├── optimization_history.json       # Optimization run history
├── optimization_schedule.json      # Scheduling state
├── eval_cache.json                # Cached evaluation dataset
└── metrics/                       # Performance metrics
    ├── query_router_optimization.json
    ├── similarity_judge_optimization.json
    └── ...
```

### State Persistence Benefits

1. **Cross-Session Learning**: Improvements persist across uvx restarts
2. **Incremental Optimization**: Builds on previous improvements
3. **Performance Tracking**: Historical metrics and trends
4. **Rollback Capability**: Can revert to previous states if needed

## Monitoring and Debugging

### Logging

Enhanced logging provides insight into system behavior:

```
INFO: Enhanced RAG with optimization enabled
INFO: DSPy optimization not due yet
INFO: Using enhanced adaptive RAG
INFO: Query intent: graph (confidence: 0.85)
INFO: Using complex query handler
INFO: Background DSPy optimization completed
```

### Status Monitoring

```python
# Get detailed status
status = searcher.get_optimization_status()

# Key metrics to monitor
- optimization_enabled: bool
- next_optimization_due: bool
- optimization_history: List[Dict]
- last_optimization_score: float
```

## Migration Guide

### From Legacy RAG

The system maintains backward compatibility:

```python
# Legacy usage still works
searcher = VaultSearcher(unified_store)
result = searcher.ask("What is machine learning?")

# Enhanced usage
enhanced_searcher = EnhancedVaultSearcher(unified_store)
result = enhanced_searcher.ask("What is machine learning?", use_enhanced=True)
```

### Configuration Migration

1. **Add DSPy state directory** to uvx environment variables
2. **Enable optimization** with `OBSIDIAN_RAG_DSPY_OPTIMIZE_ENABLED=true`
3. **Configure state persistence** path for uvx deployments
4. **Monitor initial optimization** run in logs

## Troubleshooting

### Common Issues

1. **State Directory Permissions**
   - Ensure uvx can write to configured state directory
   - Check disk space for optimization artifacts

2. **Optimization Failures**
   - Check Gemini API key and quotas
   - Verify sufficient evaluation data
   - Review optimization logs for specific errors

3. **Performance Issues**
   - Reduce `DSPY_MAX_EXAMPLES` for faster optimization
   - Use "light" auto mode for lower resource usage
   - Increase optimization interval for less frequent runs

### Debug Commands

```bash
# Check optimization status
uv run python -c "
from src.dspy_rag import EnhancedVaultSearcher
from src.unified_store import UnifiedStore
from src.config import settings

store = UnifiedStore()
searcher = EnhancedVaultSearcher(store)
print(searcher.get_optimization_status())
"

# Force optimization
uv run python -c "
from src.dspy_rag import EnhancedVaultSearcher
from src.unified_store import UnifiedStore

store = UnifiedStore()
searcher = EnhancedVaultSearcher(store)
result = searcher.force_optimization()
print(result)
"
```

## Future Enhancements

### Planned Features

1. **Ensemble Methods**: Multiple model voting for critical queries
2. **Federated Learning**: Cross-vault knowledge sharing
3. **Advanced Metrics**: Custom domain-specific evaluation
4. **Real-time Adaptation**: Online learning from user feedback
5. **Multi-modal Support**: Integration with vision and audio models

### Contributing

The DSPy enhancement system is designed to be extensible:

- **Custom Signatures**: Add domain-specific signatures in `dspy_signatures.py`
- **New Tools**: Extend ReAct agent tools in `dspy_agent.py`
- **Custom Metrics**: Add evaluation metrics in `dspy_eval.py`
- **Optimization Strategies**: Customize optimizers in `dspy_optimizer.py`

## Conclusion

The DSPy enhancements transform the Graph RAG MCP Server into a sophisticated, self-improving AI system that:

- **Adapts** to your specific vault content and query patterns
- **Learns** continuously through automated optimization
- **Reasons** through complex multi-hop relationships
- **Persists** improvements across ephemeral connections
- **Scales** with your knowledge base growth

This implementation represents a state-of-the-art approach to RAG systems, combining the power of DSPy's optimization framework with the rich graph structure of Obsidian vaults.