"""
DSPy Signatures for Enhanced Graph RAG System

This module defines specialized DSPy signatures for different types of queries and operations
in the Obsidian vault, enabling more targeted and effective prompt optimization.
"""

import dspy


class QueryIntentClassifier(dspy.Signature):
    """Classify query intent to route to optimal retrieval strategy."""
    query = dspy.InputField(desc="user query or question")
    intent = dspy.OutputField(desc="one of: semantic, graph, specific, categorical, analytical")
    confidence = dspy.OutputField(desc="confidence score 0.0-1.0")
    reasoning = dspy.OutputField(desc="brief explanation for classification")


class VaultQA(dspy.Signature):
    """Answer user questions grounded STRICTLY in provided vault snippets."""
    context = dspy.InputField(desc="relevant snippets from the vault with metadata")
    question = dspy.InputField(desc="user question")
    response = dspy.OutputField(desc="concise answer with inline citations like [#] per snippet order")


class GraphTraversal(dspy.Signature):
    """Navigate graph relationships to find connected information."""
    start_concept = dspy.InputField(desc="starting note title or concept")
    relationship_type = dspy.InputField(desc="type of relationship: links, backlinks, tags, hierarchical")
    query_context = dspy.InputField(desc="what information we're looking for")
    depth = dspy.InputField(desc="max traversal depth (1-3)")
    relevant_paths = dspy.OutputField(desc="list of traversal paths with relevance explanations")
    next_nodes = dspy.OutputField(desc="specific note IDs or concepts to explore next")


class FactExtraction(dspy.Signature):
    """Extract specific facts and properties from note content."""
    content = dspy.InputField(desc="note content or text chunk")
    fact_type = dspy.InputField(desc="type of fact to extract: date, person, concept, definition, relationship")
    query_focus = dspy.InputField(desc="specific information the user is seeking")
    extracted_facts = dspy.OutputField(desc="structured facts with confidence scores")
    source_quotes = dspy.OutputField(desc="exact quotes supporting each fact")


class AnswerWithCitations(dspy.Signature):
    """Generate comprehensive answer with specific citations to source material."""
    context_chunks = dspy.InputField(desc="retrieved chunks with note titles and metadata")
    question = dspy.InputField(desc="user question")
    retrieval_method = dspy.InputField(desc="how context was retrieved: vector, graph, hybrid")
    answer = dspy.OutputField(desc="comprehensive, well-structured answer")
    citations = dspy.OutputField(desc="specific citations: [note_title, chunk_id, supporting_quote]")
    confidence = dspy.OutputField(desc="overall confidence in answer accuracy")


class ContextExpansion(dspy.Signature):
    """Determine if additional context is needed and suggest expansion strategies."""
    current_context = dspy.InputField(desc="currently retrieved context")
    question = dspy.InputField(desc="user question")
    coverage_assessment = dspy.OutputField(desc="assessment of whether context covers the question")
    expansion_needed = dspy.OutputField(desc="yes/no whether more context is needed")
    expansion_strategy = dspy.OutputField(desc="if yes, suggest specific search terms or graph paths")


class SemanticSimilarityJudge(dspy.Signature):
    """Judge semantic similarity between query and retrieved content."""
    query = dspy.InputField(desc="user query")
    content = dspy.InputField(desc="retrieved content chunk")
    similarity_score = dspy.OutputField(desc="relevance score 0.0-1.0")
    relevance_explanation = dspy.OutputField(desc="brief explanation of relevance")


class ConceptSynthesis(dspy.Signature):
    """Synthesize information from multiple sources into coherent explanation."""
    sources = dspy.InputField(desc="multiple content sources with metadata")
    topic = dspy.InputField(desc="concept or topic to synthesize")
    synthesis = dspy.OutputField(desc="coherent explanation combining all sources")
    source_integration = dspy.OutputField(desc="how different sources complement each other")
    knowledge_gaps = dspy.OutputField(desc="what additional information might be helpful")


class RelationshipAnalysis(dspy.Signature):
    """Analyze relationships between concepts or notes in the vault."""
    concept_a = dspy.InputField(desc="first concept or note")
    concept_b = dspy.InputField(desc="second concept or note")
    context = dspy.InputField(desc="relevant content mentioning both concepts")
    relationship_type = dspy.OutputField(desc="type of relationship: causal, hierarchical, temporal, thematic")
    relationship_strength = dspy.OutputField(desc="strength of relationship 0.0-1.0")
    supporting_evidence = dspy.OutputField(desc="evidence from content supporting this relationship")


class GraphReasoningChain(dspy.Signature):
    """Multi-step reasoning through graph relationships to answer complex questions."""
    question = dspy.InputField(desc="complex question requiring multi-hop reasoning")
    graph_context = dspy.InputField(desc="current graph neighborhood information")
    reasoning_step = dspy.InputField(desc="current step in reasoning chain")
    next_action = dspy.OutputField(desc="next graph traversal or reasoning action")
    partial_answer = dspy.OutputField(desc="current understanding based on available information")
    completion_status = dspy.OutputField(desc="complete, needs_more_info, or impossible")


class ContentQualityAssessment(dspy.Signature):
    """Assess the quality and completeness of retrieved content for answering a question."""
    question = dspy.InputField(desc="user question")
    retrieved_content = dspy.InputField(desc="all retrieved content chunks")
    completeness_score = dspy.OutputField(desc="completeness score 0.0-1.0")
    quality_score = dspy.OutputField(desc="content quality score 0.0-1.0")
    missing_aspects = dspy.OutputField(desc="what key aspects are missing for complete answer")
    improvement_suggestions = dspy.OutputField(desc="suggestions for better retrieval")


class TagCategorization(dspy.Signature):
    """Categorize and organize tags for better vault organization."""
    tag = dspy.InputField(desc="tag to categorize")
    context_notes = dspy.InputField(desc="sample notes using this tag")
    category = dspy.OutputField(desc="category: project, area, resource, archive, temporal, topical")
    usage_pattern = dspy.OutputField(desc="how this tag is typically used")
    related_tags = dspy.OutputField(desc="tags that often appear together with this one")


class TemporalReasoning(dspy.Signature):
    """Reason about temporal relationships and sequences in note content."""
    content = dspy.InputField(desc="content with temporal references")
    query = dspy.InputField(desc="temporal query or question")
    timeline = dspy.OutputField(desc="extracted timeline or sequence")
    temporal_relationships = dspy.OutputField(desc="before/after/during relationships identified")
    temporal_answer = dspy.OutputField(desc="answer considering temporal aspects")