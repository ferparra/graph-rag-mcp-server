from __future__ import annotations
import sys
import logging
import frontmatter
import urllib.parse
import re
import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Literal
from fastmcp import FastMCP
from pydantic import BaseModel
# Support both package and module execution contexts
try:
    from dspy_rag import VaultSearcher, EnhancedVaultSearcher
    from unified_store import UnifiedStore
    from fs_indexer import parse_note, is_protected_test_content
    from config import settings
    from base_manager import BaseManager, BaseInfo, BaseQueryResult, BaseViewData
    from base_parser import BaseParser, BaseFile
    from cache_manager import cache_manager
    from resilience import HealthCheck, RateLimiter
    # Optional DSPy optimization imports
    try:
        from dspy_optimizer import OptimizationManager
        DSPY_OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OptimizationManager = None
        DSPY_OPTIMIZATION_AVAILABLE = False
except ImportError:  # When imported as part of a package
    from .dspy_rag import VaultSearcher, EnhancedVaultSearcher
    from .unified_store import UnifiedStore
    from .fs_indexer import parse_note, is_protected_test_content
    from .config import settings
    from .base_manager import BaseManager, BaseInfo, BaseQueryResult, BaseViewData
    from .base_parser import BaseParser, BaseFile
    from .cache_manager import cache_manager
    from .resilience import HealthCheck, RateLimiter
    # Optional DSPy optimization imports
    try:
        from .dspy_optimizer import OptimizationManager
        DSPY_OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OptimizationManager = None
        DSPY_OPTIMIZATION_AVAILABLE = False

# Configure logging to stderr to avoid corrupting MCP stdio on stdout
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("graph_rag_mcp")

class AppState:
    def __init__(self):
        # Initialize cache manager with bounded caches
        self._init_caches()
        
        # Initialize health check system
        self.health_check = HealthCheck()
        
        # Initialize rate limiter (100 requests per second)
        self.rate_limiter = RateLimiter(rate=100, burst=200)
        
        # Initialize unified store combining vector search and graph capabilities
        self.unified_store = UnifiedStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        logger.info("Connected to unified ChromaDB store: %s", settings.chroma_dir)
        
        # Register health check for ChromaDB
        self.health_check.register("chromadb", self._check_chromadb_health)
        
        # Initialize searcher with optimization capabilities if available
        if settings.dspy_optimize_enabled and DSPY_OPTIMIZATION_AVAILABLE:
            logger.info("Initializing enhanced vault searcher with DSPy optimization")
            enhanced_searcher = EnhancedVaultSearcher(unified_store=self.unified_store)
            self.searcher = enhanced_searcher  # keep as EnhancedVaultSearcher
            self._init_optimization(enhanced_searcher)
        else:
            logger.info("Using standard vault searcher (optimization disabled or unavailable)")
            # maintain attribute type by wrapping in EnhancedVaultSearcher-compatible interface
            self.searcher = EnhancedVaultSearcher(unified_store=self.unified_store)
        
        # Initialize base manager for .base file operations
        self.base_manager = BaseManager(
            unified_store=self.unified_store,
            vault_path=settings.vaults[0] if settings.vaults else None
        )
        
        # Cache cleanup task will be started when server runs
        self._cleanup_task_started = False
        self._pending_optimization: Optional[EnhancedVaultSearcher] = None
    
    def _init_caches(self):
        """Initialize application caches with appropriate bounds."""
        # Search results cache
        cache_manager.create_cache("search_results", max_size=500, ttl_seconds=600)
        
        # Note metadata cache
        cache_manager.create_cache("note_metadata", max_size=1000, ttl_seconds=1800)
        
        # Graph traversal cache
        cache_manager.create_cache("graph_traversal", max_size=200, ttl_seconds=300)
        
        # Base file cache
        cache_manager.create_cache("base_files", max_size=50, ttl_seconds=3600)
        
        logger.info("Initialized bounded caches with TTL")
    
    def _check_chromadb_health(self) -> tuple[bool, str]:
        """Health check for ChromaDB connection."""
        try:
            count = self.unified_store._collection().count()
            return True, f"Connected, {count} documents"
        except Exception as e:
            return False, f"ChromaDB error: {str(e)}"
    
    async def start_background_tasks(self):
        """Start background tasks that require an event loop."""
        if not self._cleanup_task_started:
            await cache_manager.start_cleanup_task(interval=300)
            self._cleanup_task_started = True
            logger.info("Started background cache cleanup task")
        
        # Start pending optimization if needed
        if hasattr(self, '_pending_optimization') and self._pending_optimization:
            enhanced_searcher = self._pending_optimization
            
            async def run_optimization():
                try:
                    program = enhanced_searcher.optimization_manager.get_program() if hasattr(enhanced_searcher.optimization_manager, 'get_program') else None
                    if program is not None and hasattr(enhanced_searcher.optimization_manager, 'optimizer'):
                        # Run optimization in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            enhanced_searcher.optimization_manager.optimizer.optimize_adaptive_rag,
                            program
                        )
                    logger.info("Background DSPy optimization completed")
                except Exception as e:
                    logger.error(f"Background DSPy optimization failed: {e}")
            
            # Schedule optimization as async task
            asyncio.create_task(run_optimization())
            logger.info("Scheduled background DSPy optimization")
            self._pending_optimization = None
    
    def _init_optimization(self, enhanced_searcher: EnhancedVaultSearcher):
        """Initialize DSPy optimization if enabled."""
        try:
            if getattr(enhanced_searcher, 'optimization_manager', None) is not None:
                # Check if optimization is due and run in background
                opt_mgr = enhanced_searcher.optimization_manager
                try:
                    should_run = bool(getattr(getattr(getattr(opt_mgr, 'optimizer', None), 'scheduler', None), 'should_run_optimization', lambda *_: False)("adaptive_rag"))
                except Exception:
                    should_run = False
                if should_run:
                    logger.info("DSPy optimization is due - will schedule when server starts")
                    self._pending_optimization = enhanced_searcher
                else:
                    logger.info("DSPy optimization not due yet")
                    self._pending_optimization = None
                    
                # Log optimization status
                status = enhanced_searcher.get_optimization_status()
                logger.info(f"DSPy optimization status: {status.get('optimization_enabled', False)}")
                
        except Exception as e:
            logger.error(f"Failed to initialize DSPy optimization: {e}")

mcp = FastMCP("Graph-RAG Obsidian Server")

# Global variable for app state
app_state: Optional[AppState] = None
_initialization_lock = asyncio.Lock()

async def get_app_state() -> AppState:
    """Get or create the app state, ensuring background tasks are started."""
    global app_state
    
    async with _initialization_lock:
        if app_state is None:
            app_state = AppState()
            # Start background tasks now that we have an event loop
            await app_state.start_background_tasks()
        
    return app_state

class SmartSearchEngine:
    """Intelligent search engine that routes queries to optimal search strategies."""
    
    def __init__(self, unified_store: UnifiedStore, searcher: VaultSearcher):
        self.unified_store = unified_store
        self.searcher = searcher
        
        # Patterns for query intent detection
        self.tag_patterns = re.compile(r'#(\w+)|tag[s]?\s*[:=]\s*(\w+)|tagged\s+with\s+(\w+)', re.IGNORECASE)
        self.relationship_patterns = re.compile(r'link[s]?\s+to|connect[s]?\s+to|related\s+to|references?|mentions?|backlink[s]?|graph|network', re.IGNORECASE)
        self.specific_patterns = re.compile(r'in\s+note\s+|from\s+file\s+|in\s+\w+\s+document|in\s+document|specific[ally]*|exact[ly]*', re.IGNORECASE)
        self.semantic_patterns = re.compile(r'about|concept|idea|topic|understand|explain|meaning|definition', re.IGNORECASE)
    
    def analyze_query_intent(self, query: str) -> QueryIntent:
        """Analyze query to determine user intent and optimal search strategy."""
        
        # Extract potential entities (simple approach)
        entities = []
        
        # Check for explicit tags
        tag_matches = self.tag_patterns.findall(query)
        if tag_matches:
            for match in tag_matches:
                entities.extend([tag for tag in match if tag])
        
        # Determine intent type and confidence
        intent_scores = {
            "categorical": 0.0,  # Tag-based search
            "relationship": 0.0,  # Graph traversal
            "specific": 0.0,     # Targeted search
            "semantic": 0.0      # Vector similarity
        }
        
        # Score based on patterns
        if self.tag_patterns.search(query):
            intent_scores["categorical"] += 0.8
        
        if self.relationship_patterns.search(query):
            intent_scores["relationship"] += 0.7
            
        if self.specific_patterns.search(query):
            intent_scores["specific"] += 0.6
            
        if self.semantic_patterns.search(query):
            intent_scores["semantic"] += 0.6
        
        # Default semantic search for general queries
        if max(intent_scores.values()) < 0.3:
            intent_scores["semantic"] = 0.8
        
        # Determine primary intent
        primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        confidence = intent_scores[primary_intent]
        
        # Map intent to strategy
        strategy_mapping = {
            "categorical": "tag",
            "relationship": "graph", 
            "specific": "vector",
            "semantic": "vector"
        }
        
        # Use hybrid for complex queries or low confidence
        suggested_strategy = "hybrid" if confidence < 0.6 or len(entities) > 0 and intent_scores["semantic"] > 0.4 else strategy_mapping[primary_intent]
        
        return QueryIntent(
            intent_type=primary_intent,
            confidence=confidence,
            extracted_entities=entities,
            suggested_strategy=suggested_strategy
        )
    
    def generate_chunk_uri(self, chunk_metadata: Dict[str, Any], vault_name: str = "My Vault") -> str:
        """Generate Obsidian URI for a specific chunk."""
        note_path = chunk_metadata.get('path', '')
        header_text = chunk_metadata.get('header_text', '')
        
        # Remove vault path prefix to get relative path
        relative_path = note_path
        if '/' in note_path:
            relative_path = note_path.split('/')[-1]
        
        # Remove .md extension
        if relative_path.endswith('.md'):
            relative_path = relative_path[:-3]
        
        # URL-encode the file component for spaces and special characters
        encoded_file = urllib.parse.quote(relative_path, safe="")
        base_uri = f"obsidian://open?vault={vault_name}&file={encoded_file}"
        
        # Add header anchor if available
        if header_text:
            # Convert header to URL-safe anchor
            anchor = urllib.parse.quote(header_text.lower(), safe="")
            base_uri += f"#{anchor}"
        
        return base_uri
    
    def enhance_results_with_uris(self, hits: List[Dict], vault_name: str = "My Vault") -> List[Dict]:
        """Add URIs and enhanced metadata to search results."""
        enhanced_hits = []
        
        for hit in hits:
            enhanced_hit = hit.copy()
            meta = hit.get('meta', {})
            
            # Generate chunk URI
            enhanced_hit['chunk_uri'] = self.generate_chunk_uri(meta, vault_name)
            
            # Add chunk-level metadata
            chunk_info = {
                'chunk_id': meta.get('chunk_id', ''),
                'chunk_type': meta.get('chunk_type', ''),
                'importance_score': meta.get('importance_score', 0.5),
                'header_context': meta.get('parent_headers', ''),
                'retrieval_method': hit.get('retrieval_method', 'vector_search')
            }
            enhanced_hit['chunk_info'] = chunk_info
            
            # Add note-level metadata  
            note_info = {
                'note_id': meta.get('note_id', ''),
                'title': meta.get('title', ''),
                'tags': meta.get('tags', '').split(',') if meta.get('tags') else [],
                'links_to': meta.get('links_to', '').split(',') if meta.get('links_to') else []
            }
            enhanced_hit['note_info'] = note_info
            
            enhanced_hits.append(enhanced_hit)
        
        return enhanced_hits
    
    def smart_search(self, query: str, k: int = 6, vault_filter: Optional[str] = None) -> SmartSearchResult:
        """Perform intelligent search based on query analysis."""
        
        # Analyze query intent
        intent = self.analyze_query_intent(query)
        strategy = intent.suggested_strategy
        
        where_clause = {}
        if vault_filter:
            where_clause["vault"] = {"$eq": vault_filter}
        
        hits = []
        explanation = ""
        related_chunks = []
        
        try:
            if strategy == "tag":
                # Tag-based search with fuzzy matching
                hits = self._tag_search(query, intent.extracted_entities, k, where_clause)
                explanation = f"Used tag-based search for categorical query. Found content tagged with: {', '.join(intent.extracted_entities)}"
                
            elif strategy == "graph":
                # Graph traversal search
                hits, related_chunks = self._graph_search(query, k, where_clause)
                explanation = "Used graph traversal to find related content through links and relationships"
                
            elif strategy == "vector":
                # Vector similarity search
                hits = self.searcher.search(query, k=k, where=where_clause if where_clause else None)
                explanation = "Used semantic vector search for conceptual query matching"
                
            elif strategy == "hybrid":
                # Hybrid approach combining multiple strategies
                hits, related_chunks = self._hybrid_search(query, intent, k, where_clause)
                explanation = "Used hybrid search combining semantic similarity and graph relationships"
                
        except Exception as e:
            logger.error(f"Smart search error with strategy {strategy}: {e}")
            # Fallback to basic vector search
            hits = self.searcher.search(query, k=k, where=where_clause if where_clause else None)
            strategy = "vector"
            explanation = f"Fallback to vector search due to error: {str(e)}"
        
        # Enhance results with URIs and metadata
        enhanced_hits = self.enhance_results_with_uris(hits)
        if related_chunks:
            enhanced_related = self.enhance_results_with_uris(related_chunks)
        else:
            enhanced_related = None
        
        return SmartSearchResult(
            query=query,
            strategy_used=strategy,
            hits=enhanced_hits,
            total_results=len(enhanced_hits),
            explanation=explanation,
            related_chunks=enhanced_related
        )
    
    def _tag_search(self, query: str, entities: List[str], k: int, where_clause: Dict) -> List[Dict]:
        """Enhanced tag search with fuzzy matching."""
        # If no entities extracted, try to extract from query
        if not entities:
            # Simple extraction of potential tag terms
            words = re.findall(r'\b\w+\b', query.lower())
            entities = [word for word in words if len(word) > 2][:3]  # Limit to 3 entities
        
        # Use fuzzy tag matching from unified store
        return self.unified_store.fuzzy_tag_search(entities, k=k, where=where_clause)
    
    def _graph_search(self, query: str, k: int, where_clause: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Graph-based search through relationships."""
        # First get initial semantic matches
        initial_hits = self.searcher.search(query, k=max(3, k//2), where=where_clause if where_clause else None)
        
        if not initial_hits:
            return [], []
        
        # Expand using graph relationships
        related_chunks = []
        
        for hit in initial_hits:
            chunk_id = hit.get('meta', {}).get('chunk_id', '')
            if chunk_id:
                neighbors = self.unified_store.get_chunk_neighbors(chunk_id)
                for neighbor in neighbors[:2]:  # Limit neighbors per hit
                    related_chunks.append({
                        'chunk_id': neighbor['chunk_id'],
                        'relationship': neighbor['relationship'],
                        'text': f"[Related via {neighbor['relationship']}]",
                        'meta': {
                            'chunk_type': neighbor.get('chunk_type', ''),
                            'importance_score': neighbor.get('importance_score', 0.5)
                        }
                    })
        
        # Combine initial hits with top related chunks
        all_hits = initial_hits + related_chunks[:k-len(initial_hits)]
        
        return all_hits[:k], related_chunks
    
    def _hybrid_search(self, query: str, intent: QueryIntent, k: int, where_clause: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Hybrid search combining vector + graph + tag approaches."""
        all_hits = []
        related_chunks = []
        
        # Vector search (primary)
        vector_hits = self.searcher.search(query, k=max(3, k//2), where=where_clause if where_clause else None)
        all_hits.extend(vector_hits)
        
        # Tag search if entities detected
        if intent.extracted_entities:
            tag_hits = self.unified_store.fuzzy_tag_search(intent.extracted_entities, k=2, where=where_clause)
            all_hits.extend(tag_hits)
        
        # Graph expansion for top results
        if vector_hits:
            top_chunk_id = vector_hits[0].get('meta', {}).get('chunk_id', '')
            if top_chunk_id:
                neighbors = self.unified_store.get_chunk_neighbors(top_chunk_id, include_sequential=True, include_hierarchical=True)
                related_chunks.extend(neighbors[:3])
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_hits = []
        for hit in all_hits:
            hit_id = hit.get('id', '') or hit.get('chunk_id', '')
            if hit_id and hit_id not in seen_ids:
                seen_ids.add(hit_id)
                unique_hits.append(hit)
        
        return unique_hits[:k], related_chunks[:k]

# Helper function for safe ChromaDB result access
def safe_get_chroma_results(results: Any, index: int = 0):
    """Safely extract metadata and document from ChromaDB results."""
    metadatas = results.get('metadatas')
    documents = results.get('documents')
    
    metadata = metadatas[index] if metadatas and len(metadatas) > index else {}
    document = documents[index] if documents and len(documents) > index else ""
    
    return metadata, document

# Initialize smart search engine with a VaultSearcher-compatible object
# Smart search engine will be created when needed
smart_search_engine: Optional[SmartSearchEngine] = None

async def get_smart_search_engine() -> SmartSearchEngine:
    """Get or create the smart search engine."""
    global smart_search_engine
    
    if smart_search_engine is None:
        state = await get_app_state()
        smart_search_engine = SmartSearchEngine(state.unified_store, VaultSearcher(state.unified_store))
    
    return smart_search_engine

class SearchResult(BaseModel):
    hits: List[Dict]
    total_results: int
    query: str

class AnswerResult(BaseModel):
    question: str
    answer: str
    context: str
    success: bool

class NoteInfo(BaseModel):
    id: str
    title: str
    path: str
    tags: List[str]
    links: List[str]
    content: str
    frontmatter: Dict

class GraphResult(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
    stats: Optional[Dict] = None

class SmartSearchResult(BaseModel):
    query: str
    strategy_used: Literal["vector", "graph", "tag", "hybrid"]
    hits: List[Dict]
    total_results: int
    explanation: str
    related_chunks: Optional[List[Dict]] = None

class ChunkContext(BaseModel):
    chunk_id: str
    chunk_uri: str
    context_chunks: List[Dict]
    relationships: List[Dict]
    note_title: str
    path: str

class QueryIntent(BaseModel):
    intent_type: Literal["semantic", "relationship", "categorical", "specific"]
    confidence: float
    extracted_entities: List[str]
    suggested_strategy: Literal["vector", "graph", "tag", "hybrid"]

@mcp.tool()
async def smart_search(
    query: str,
    k: int = 6,
    vault_filter: Optional[str] = None
) -> SmartSearchResult:
    """Intelligent search that analyzes query intent and chooses optimal search strategy.
    
    This replaces basic search with smart routing between:
    - Vector similarity for semantic queries
    - Graph traversal for relationship queries  
    - Tag matching for categorical queries
    - Hybrid approach for complex queries
    
    Returns enhanced results with proper Obsidian URIs for chunks.
    """
    engine = await get_smart_search_engine()
    return engine.smart_search(query, k=k, vault_filter=vault_filter)

@mcp.tool()
async def search_notes(
    query: str,
    k: int = 6,
    vault_filter: Optional[str] = None,
    tag_filter: Optional[str] = None
) -> SearchResult:
    """Basic vector search across vault chunks using ChromaDB.
    
    Note: Consider using smart_search instead for better results with intelligent routing.
    """
    state = await get_app_state()
    where = {}
    if vault_filter:
        where["vault"] = {"$eq": vault_filter}
    if tag_filter:
        # Use improved tag matching
        tag_results = state.unified_store.fuzzy_tag_search([tag_filter], k=k, where=where)
        return SearchResult(
            hits=tag_results,
            total_results=len(tag_results),
            query=query
        )
    
    where_clause = where if where else None
    hits = state.searcher.search(query, k=k, where=where_clause)
    
    return SearchResult(
        hits=hits,
        total_results=len(hits),
        query=query
    )

@mcp.tool()
async def answer_question(
    question: str,
    vault_filter: Optional[str] = None,
    tag_filter: Optional[str] = None,
    use_smart_search: bool = True
) -> AnswerResult:
    """Enhanced RAG-powered Q&A using intelligent search routing and Gemini 2.5 Flash.
    
    Uses smart search routing to find the most relevant content through:
    - Vector similarity for semantic questions
    - Graph traversal for relationship questions
    - Tag matching for categorical questions
    - Hybrid approach for complex questions
    """
    state = await get_app_state()
    where = {}
    if vault_filter:
        where["vault"] = {"$eq": vault_filter}
    
    # Use smart search if enabled and no specific tag filter
    if use_smart_search and not tag_filter:
        try:
            # Get smart search results
            engine = await get_smart_search_engine()
            smart_results = engine.smart_search(question, k=8, vault_filter=vault_filter)
            
            # Format context for RAG with enhanced information
            ctx_parts = []
            for i, hit in enumerate(smart_results.hits, start=1):
                title = hit['note_info']['title']
                chunk_uri = hit['chunk_uri']
                text = hit['text']
                chunk_info = hit['chunk_info']
                
                # Add retrieval method and relationship info
                retrieval_info = f"[{chunk_info['retrieval_method']}]"
                if chunk_info.get('header_context'):
                    retrieval_info += f" {chunk_info['header_context']}"
                
                ctx_parts.append(f"[{i}] {title} {retrieval_info}\nURI: {chunk_uri}\n{text}\n")
            
            context = "\n".join(ctx_parts) if ctx_parts else "NO CONTEXT FOUND"
            
            # Generate answer using the RAG system - use the existing ask method
            try:
                result = await state.searcher.ask(question)
                enhanced_answer = result.get('answer', 'No answer generated')
                search_explanation = f"\n\n[Search Strategy: {smart_results.strategy_used} - {smart_results.explanation}]"
                
                return AnswerResult(
                    question=question,
                    answer=enhanced_answer + search_explanation,
                    context=context,
                    success=True
                )
            except Exception as e:
                print(f"Smart RAG error: {e}")
            
            # Fallback response with smart search context
            return AnswerResult(
                question=question,
                answer=f"Based on smart search ({smart_results.strategy_used}), I found relevant information but couldn't generate a complete answer. Please review the context provided.",
                context=context,
                success=True
            )
            
        except Exception as e:
            print(f"Smart search error in answer_question: {e}")
            # Fall back to traditional search
    
    # Traditional search path
    if tag_filter:
        # Use improved tag matching
        tag_results = state.unified_store.fuzzy_tag_search([tag_filter], k=6, where=where)
        
        # Format for traditional RAG
        ctx_parts = []
        for i, hit in enumerate(tag_results, start=1):
            title = hit['meta'].get('title', 'Unknown')
            path = hit['meta'].get('path', 'Unknown')
            text = hit['text']
            ctx_parts.append(f"[{i}] {title} ({path})\n{text}\n")
        
        context = "\n".join(ctx_parts) if ctx_parts else "NO CONTEXT FOUND"
        
        try:
            result = await state.searcher.ask(question)
            return AnswerResult(
                question=question,
                answer=result.get('answer', 'No answer generated'),
                context=context,
                success=True
            )
        except Exception as e:
            print(f"Tag-based RAG error: {e}")
        
        return AnswerResult(
            question=question,
            answer="Found relevant tagged content but couldn't generate a complete answer.",
            context=context,
            success=True
        )
    
    # Standard vector search fallback
    where_clause = where if where else None
    result = await state.searcher.ask(question, where=where_clause)
    
    return AnswerResult(**result)

@mcp.tool()
async def graph_neighbors(
    note_id_or_title: str,
    depth: int = 1,
    relationship_types: Optional[List[str]] = None
) -> GraphResult:
    """Get neighboring notes in the graph up to specified depth."""
    state = await get_app_state()
    neighbors = state.unified_store.get_neighbors(
        note_id_or_title, 
        depth=depth, 
        relationship_types=relationship_types
    )
    return GraphResult(
        nodes=neighbors,
        edges=[],
        stats={"neighbor_count": len(neighbors)}
    )

@mcp.tool()
async def get_subgraph(
    seed_notes: List[str],
    depth: int = 1
) -> GraphResult:
    """Get a subgraph containing seed notes and their neighbors."""
    state = await get_app_state()
    subgraph = state.unified_store.get_subgraph(seed_notes, depth)
    return GraphResult(**subgraph)

@mcp.tool()
async def list_notes(
    limit: Optional[int] = 50,
    vault_filter: Optional[str] = None
) -> List[Dict]:
    """List all notes in the vault with metadata."""
    state = await get_app_state()
    notes = state.unified_store.get_all_notes(limit=limit)
    
    if vault_filter:
        notes = [n for n in notes if n.get("meta", {}).get("vault") == vault_filter]
    
    return notes

def _load_note(note_path: str) -> NoteInfo:
    """Helper function to load a note and return NoteInfo."""
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    note = parse_note(path)
    
    return NoteInfo(
        id=note.id,
        title=note.title,
        path=str(note.path),
        tags=note.tags,
        links=note.links,
        content=note.text,
        frontmatter=note.frontmatter
    )

@mcp.tool()
async def read_note(note_path: str) -> NoteInfo:
    """Read the full content of a note by path."""
    return _load_note(note_path)

@mcp.tool()
async def get_note_properties(note_path: str) -> Dict:
    """Get frontmatter properties of a note."""
    note_info = _load_note(note_path)
    return note_info.frontmatter

@mcp.tool()
async def update_note_properties(
    note_path: str,
    properties: Dict[str, Any],
    merge: bool = True
) -> Dict:
    """Update frontmatter properties of a note."""
    state = await get_app_state()
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)
    
    if merge:
        post.metadata.update(properties)
    else:
        post.metadata = properties
    
    with open(path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)
    
    note = parse_note(path)
    state.unified_store.upsert_note(note)
    
    return post.metadata

@mcp.tool()
async def archive_note(
    note_path: str,
    archive_folder: Optional[str] = None
) -> str:
    """Move a note to the archive folder."""
    state = await get_app_state()
    archive_name = archive_folder or settings.archive_folder
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    vault_root = None
    for vault in settings.vaults:
        try:
            path.relative_to(vault)
            vault_root = vault
            break
        except ValueError:
            continue
    
    if not vault_root:
        raise ValueError(f"Note {note_path} not in any configured vault")
    
    archive_dir = vault_root / archive_name
    archive_dir.mkdir(exist_ok=True)
    
    new_path = archive_dir / path.name
    path.rename(new_path)
    
    state.unified_store.delete_note(str(path.relative_to(vault_root)))
    
    return str(new_path)

@mcp.tool()
async def create_note(
    title: str,
    content: str = "",
    folder: Optional[str] = None,
    tags: Optional[List[str]] = None,
    para_type: Optional[str] = None,
    enrich: bool = True
) -> Dict[str, Any]:
    state = await get_app_state()
    """Create a new Obsidian note with enriched frontmatter.
    
    Args:
        title: Note title (will be used as filename)
        content: Initial note content (markdown)
        folder: Folder path within vault (e.g., "Projects", "00 Inbox")
        tags: Initial tags to add
        para_type: PARA type hint (project/area/resource/archive)
        enrich: Whether to apply AI enrichment for PARA classification
    
    Returns:
        Dict with created note path and metadata
    """
    import frontmatter
    from datetime import datetime
    
    # Sanitize title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"{safe_title}.md"
    
    # Determine folder
    vault_root = settings.vaults[0]
    if folder:
        note_folder = vault_root / folder
        note_folder.mkdir(parents=True, exist_ok=True)
    else:
        # Default to inbox if it exists, otherwise root
        inbox = vault_root / "00 Inbox"
        if inbox.exists():
            note_folder = inbox
        else:
            note_folder = vault_root
    
    note_path = note_folder / filename
    
    # Check if file already exists
    if note_path.exists():
        # Add timestamp to make unique
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename: str = f"{safe_title}_{timestamp}.md"
        note_path: Path = note_folder / filename
    
    # Prepare single, valid frontmatter by merging any YAML blocks from content
    # and tool-provided defaults/overrides into one block.
    import re
    import yaml

    now_iso = datetime.now().isoformat()

    # Helper: normalize tags to a list[str]
    def _normalize_tags(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.lstrip('#')]
        if isinstance(value, list):
            out: list[str] = []
            for t in value:
                if isinstance(t, str):
                    out.append(t.lstrip('#'))
            return out
        return []

    # Helper: merge two tag lists with order preserved and no duplicates
    def _merge_tags(a: list[str], b: list[str]) -> list[str]:
        seen = set()
        merged: list[str] = []
        for t in (a + b):
            if t not in seen:
                seen.add(t)
                merged.append(t)
        return merged

    # Start by parsing any frontmatter already present in the content
    base_post = frontmatter.loads(content)
    merged_meta: Dict[str, Any] = dict(base_post.metadata or {})
    body: str = base_post.content or ""

    # Additionally, some content may (incorrectly) contain another YAML block
    # immediately after the first one. Merge any additional leading YAML blocks.
    yaml_block_pattern = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
    while True:
        m = yaml_block_pattern.match(body)
        if not m:
            break
        block_text = m.group(1)
        try:
            loaded = yaml.safe_load(block_text)
        except Exception:
            # Not valid YAML; treat as content
            break
        if isinstance(loaded, dict) and loaded:
            # Merge this block and remove it from the body
            for k, v in loaded.items():
                if k == "tags":
                    existing = _normalize_tags(merged_meta.get("tags"))
                    merged_meta["tags"] = _merge_tags(existing, _normalize_tags(v))
                else:
                    # Prefer earlier keys unless we explicitly override later
                    if k not in merged_meta:
                        merged_meta[k] = v
            body = body[m.end():]
        else:
            # Empty or non-dict YAML (likely a horizontal rule usage) — stop
            break

    # Defaults: created/modified timestamps
    if "created" not in merged_meta:
        merged_meta["created"] = now_iso
    # Always set modified to now for a newly created note
    merged_meta["modified"] = now_iso

    # Merge user-provided tags and PARA info
    if tags:
        merged_meta["tags"] = _merge_tags(_normalize_tags(merged_meta.get("tags")), _normalize_tags(tags))
    # PARA type + tag
    if para_type and para_type in ["project", "area", "resource", "archive"]:
        merged_meta["para_type"] = para_type
        merged_meta["tags"] = _merge_tags(_normalize_tags(merged_meta.get("tags")), [f"para/{para_type}"])

    # Create the note with a single consolidated frontmatter block
    post = frontmatter.Post(body, handler=None, **merged_meta)
    
    # Write the initial note
    with open(note_path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)
    
    # Index the note
    note = parse_note(note_path)
    state.unified_store.upsert_note(note)
    
    # Apply enrichment if requested
    enriched_metadata = {}
    if enrich and content.strip():  # Only enrich if there's content
        try:
            # Import enrichment module dynamically since it's in scripts directory
            import importlib.util
            script_path = Path(__file__).parent.parent / "scripts" / "enrich_para_taxonomy.py"
            spec = importlib.util.spec_from_file_location("enrich_para_taxonomy", script_path)
            ParaTaxonomyEnricher = None
            if spec and spec.loader:
                enrich_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(enrich_module)
                ParaTaxonomyEnricher = enrich_module.ParaTaxonomyEnricher
            
            # Initialize enricher
            if ParaTaxonomyEnricher is None:
                raise ImportError("Could not import ParaTaxonomyEnricher")
            enricher = ParaTaxonomyEnricher()
            
            # Enrich the newly created note
            result = enricher.enrich_note_properties(str(note_path), dry_run=False)
            
            if result:
                enriched_metadata = result.get('enriched_properties', {})
                
                # Re-index after enrichment
                updated_note = parse_note(note_path)
                state.unified_store.upsert_note(updated_note)
        except Exception as e:
            # Enrichment failed, but note was still created
            enriched_metadata: dict[str, str] = {"enrichment_error": str(e)}
    
    # Return created note info
    final_metadata = dict(merged_meta)
    if enriched_metadata:
        final_metadata.update(enriched_metadata)
    
    return {
        "path": str(note_path),
        "title": title,
        "folder": str(note_folder.relative_to(vault_root)),
        "metadata": final_metadata,
        "enriched": enrich and bool(enriched_metadata),
        "message": f"Note created: {note_path.name}"
    }

@mcp.tool()
async def create_folder(folder_path: str, vault_name: Optional[str] = None) -> str:
    """Create a new folder in the vault."""
    if vault_name:
        vault_root = None
        for vault in settings.vaults:
            if vault.name == vault_name:
                vault_root = vault
                break
        if not vault_root:
            raise ValueError(f"Vault {vault_name} not found")
    else:
        vault_root = settings.vaults[0]
    
    folder = vault_root / folder_path
    folder.mkdir(parents=True, exist_ok=True)
    
    return str(folder)

@mcp.tool()
async def add_content_to_note(
    note_path: str,
    content: str,
    position: str = "end"
) -> str:
    """Add content to an existing note."""
    state = await get_app_state()
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)
    
    if position == "start":
        post.content = content + "\n\n" + post.content
    else:
        post.content = post.content + "\n\n" + content
    
    with open(path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)
    
    note = parse_note(path)
    state.unified_store.upsert_note(note)
    
    return f"Content added to {note_path}"

@mcp.tool()
async def update_note_section(
    note_path: str,
    section_heading: str,
    new_content: str,
    heading_level: Optional[int] = None
) -> Dict[str, Any]:
    state = await get_app_state()
    """Replace the content of a specific Markdown section in-place.

    This updates only the body of a section identified by its heading, keeping the
    heading itself intact. The rest of the note remains unchanged. If the section
    is not found, an error is raised. Uses ATX-style headings ("#", "##", etc.).

    Args:
        note_path: Path to the note (absolute or vault-relative).
        section_heading: The visible heading text to match (case-insensitive).
        new_content: Replacement body for the section (Markdown). Inserted in-place.
        heading_level: Optional exact heading level to match (1-6). If omitted, the
            first matching heading text is used regardless of level.

    Returns:
        Dict with path, section name, and a short message.
    """
    import re

    path = Path(note_path)
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")

    # Load note preserving frontmatter
    with open(path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)

    content = post.content
    lines = content.splitlines(keepends=True)

    # Normalize function for comparing heading text
    def norm(s: str) -> str:
        return " ".join(s.strip().split()).lower()

    target_text = norm(section_heading)

    # Simple ATX heading parser, ignoring fenced code blocks
    def parse_heading(line: str) -> Optional[tuple[int, str]]:
        s = line.strip()
        if not s.startswith('#'):
            return None
        # Count leading '#'
        i = 0
        while i < len(s) and s[i] == '#':
            i += 1
        if i == 0 or i > 6:
            return None
        title = s[i:].strip()
        # Strip optional trailing hashes
        title = re.sub(r"\s*#+\s*$", "", title).strip()
        if not title:
            return None
        return i, title

    # Track fenced code blocks to avoid false positive headings
    in_fence = False
    fence_re = re.compile(r"^\s*```")

    headings: list[tuple[int, int, int, str]] = []  # (line_index, level, char_index_start, title)
    char_index = 0
    for idx, line in enumerate(lines):
        if fence_re.match(line):
            in_fence = not in_fence
        if not in_fence:
            parsed = parse_heading(line)
            if parsed:
                lvl, title = parsed
                headings.append((idx, lvl, char_index, title))
        char_index += len(line)

    # Find target heading index in lines
    target_idx: Optional[int] = None
    target_lvl: Optional[int] = None

    for idx, lvl, _start_char, title in headings:
        if norm(title) == target_text and (heading_level is None or lvl == heading_level):
            target_idx = idx
            target_lvl = lvl
            break

    if target_idx is None or target_lvl is None:
        raise ValueError(f"Section heading not found: '{section_heading}'")

    # Determine the end of the section: next heading with level <= target level, or EOF
    end_idx = len(lines)
    for idx, lvl, _start_char, _title in headings:
        if idx <= target_idx:
            continue
        if lvl <= target_lvl:
            end_idx = idx
            break

    # Build new content: keep heading line, replace body between heading and end_idx
    prefix = ''.join(lines[: target_idx + 1])
    suffix = ''.join(lines[end_idx:])

    body = new_content.rstrip('\n')
    # Ensure one blank line after the heading before body (common Markdown style)
    if body:
        replacement = prefix + "\n" + body + "\n" + suffix
    else:
        # Empty body: keep a single blank line after heading for readability
        replacement = prefix + "\n" + suffix

    post.content = replacement

    # Persist changes
    with open(path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)

    # Re-index updated note
    note = parse_note(path)
    state.unified_store.upsert_note(note)

    return {
        "path": str(path),
        "section": section_heading,
        "message": "Section content updated in-place"
    }

@mcp.tool()
async def get_backlinks(note_id_or_path: str) -> List[Dict]:
    """Get all notes that link to the specified note."""
    state = await get_app_state()
    return state.unified_store.get_backlinks(note_id_or_path)

@mcp.tool()
async def get_notes_by_tag(tag: str) -> List[Dict]:
    """Get all notes that have the specified tag."""
    state = await get_app_state()
    return state.unified_store.get_notes_by_tag(tag)

@mcp.tool()
async def traverse_from_chunk(
    chunk_id: str,
    max_depth: int = 2,
    include_sequential: bool = True,
    include_hierarchical: bool = True,
    include_content_links: bool = True
) -> ChunkContext:
    """Traverse the graph starting from a specific semantic chunk.
    
    This provides chunk-level graph navigation, allowing you to explore
    relationships from any specific piece of content in the vault.
    """
    state = await get_app_state()
    try:
        # Get the source chunk details
        col = state.unified_store._collection()
        source_results = col.get(
            where={"chunk_id": {"$eq": chunk_id}},
            include=['metadatas', 'documents']
        )
        
        source_meta, source_doc = safe_get_chroma_results(source_results)
        if not source_meta:
            raise ValueError(f"Chunk not found: {chunk_id}")
        
        # Generate URI for source chunk
        engine = await get_smart_search_engine()
        chunk_uri = engine.generate_chunk_uri(dict(source_meta))
        
        # Collect related chunks through multiple relationship types
        all_related = []
        seen_chunks = {chunk_id}
        current_level = [chunk_id]
        
        for depth in range(max_depth):
            next_level = []
            
            for current_chunk in current_level:
                # Get chunk neighbors
                neighbors = state.unified_store.get_chunk_neighbors(
                    current_chunk,
                    include_sequential=include_sequential,
                    include_hierarchical=include_hierarchical
                )
                
                for neighbor in neighbors:
                    neighbor_id = neighbor['chunk_id']
                    if neighbor_id not in seen_chunks:
                        seen_chunks.add(neighbor_id)
                        next_level.append(neighbor_id)
                        
                        # Get neighbor content
                        neighbor_results = col.get(
                            where={"chunk_id": {"$eq": neighbor_id}},
                            include=['metadatas', 'documents']
                        )
                        
                        neighbor_meta, neighbor_doc = safe_get_chroma_results(neighbor_results)
                        if neighbor_meta:
                            
                            engine = await get_smart_search_engine()
                            all_related.append({
                                'chunk_id': neighbor_id,
                                'chunk_uri': engine.generate_chunk_uri(dict(neighbor_meta)),
                                'relationship': neighbor['relationship'],
                                'depth': depth + 1,
                                'content': neighbor_doc[:200] + "..." if len(neighbor_doc) > 200 else neighbor_doc,
                                'chunk_type': neighbor_meta.get('chunk_type', ''),
                                'header_text': neighbor_meta.get('header_text', ''),
                                'importance_score': neighbor_meta.get('importance_score', 0.5),
                                'note_title': neighbor_meta.get('title', ''),
                                'path': neighbor_meta.get('path', '')
                            })
                
                # Get content-based links if requested
                if include_content_links and depth == 0:  # Only for first level to avoid explosion
                    links_to = source_meta.get('links_to', '')
                    if links_to:
                        linked_notes = state.unified_store._parse_delimited_string(str(links_to))
                        for linked_note in linked_notes[:3]:  # Limit to avoid too many results
                            # Find chunks in linked notes
                            linked_chunks = col.get(
                                where={"note_id": {"$eq": linked_note}},
                                include=['metadatas', 'documents'],
                                limit=2  # Top chunks from linked notes
                            )
                            
                            linked_metadatas = linked_chunks.get('metadatas')
                            if linked_metadatas:
                                for i, linked_meta in enumerate(linked_metadatas):
                                    linked_chunk_id = str(linked_meta.get('chunk_id', ''))
                                    if linked_chunk_id and linked_chunk_id not in seen_chunks:
                                        seen_chunks.add(linked_chunk_id)
                                        _, linked_doc = safe_get_chroma_results(linked_chunks, i)
                                        
                                        engine = await get_smart_search_engine()
                                        all_related.append({
                                            'chunk_id': linked_chunk_id,
                                            'chunk_uri': engine.generate_chunk_uri(dict(linked_meta)),
                                            'relationship': 'content_link',
                                            'depth': 1,
                                            'content': linked_doc[:200] + "..." if len(linked_doc) > 200 else linked_doc,
                                            'chunk_type': linked_meta.get('chunk_type', ''),
                                            'header_text': linked_meta.get('header_text', ''),
                                            'importance_score': linked_meta.get('importance_score', 0.5),
                                            'note_title': linked_meta.get('title', ''),
                                            'path': linked_meta.get('path', '')
                                        })
            
            current_level = next_level
            if not current_level:
                break
        
        # Create relationship summary
        relationships = []
        relationship_counts = {}
        for chunk in all_related:
            rel_type = chunk['relationship']
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        for rel_type, count in relationship_counts.items():
            relationships.append({
                'type': rel_type,
                'count': count,
                'description': _get_relationship_description(rel_type)
            })
        
        return ChunkContext(
            chunk_id=chunk_id,
            chunk_uri=chunk_uri,
            context_chunks=all_related,
            relationships=relationships,
            note_title=source_meta.get('title', ''),
            path=source_meta.get('path', '')
        )
        
    except Exception as e:
        logger.error(f"Error traversing from chunk {chunk_id}: {e}")
        raise

@mcp.tool()
async def get_related_chunks(
    query: str,
    chunk_types: Optional[List[str]] = None,
    max_results: int = 10,
    min_importance: float = 0.3
) -> List[Dict]:
    """Find chunks related to a query through graph relationships.
    
    Unlike vector search, this finds chunks connected through the graph
    structure (links, backlinks, hierarchical relationships).
    """
    state: AppState = await get_app_state()
    try:
        # First get initial vector matches
        initial_hits = state.searcher.search(query, k=5)
        
        if not initial_hits:
            return []
        
        related_chunks = []
        seen_chunks = set()
        
        for hit in initial_hits:
            source_chunk_id = hit.get('meta', {}).get('chunk_id', '')
            if not source_chunk_id:
                continue
                
            # Get neighbors for this chunk
            neighbors = state.unified_store.get_chunk_neighbors(source_chunk_id)
            
            for neighbor in neighbors:
                neighbor_id = neighbor['chunk_id']
                importance = neighbor.get('importance_score', 0.5)
                
                if (neighbor_id not in seen_chunks and 
                    importance >= min_importance):
                    
                    seen_chunks.add(neighbor_id)
                    
                    # Filter by chunk type if specified
                    if chunk_types and neighbor.get('chunk_type', '') not in chunk_types:
                        continue
                    
                    # Get full chunk content
                    col = state.unified_store._collection()
                    chunk_results = col.get(
                        where={"chunk_id": {"$eq": neighbor_id}},
                        include=['metadatas', 'documents']
                    )
                    
                    chunk_meta, chunk_doc = safe_get_chroma_results(chunk_results)
                    if chunk_meta:
                        
                        engine = await get_smart_search_engine()
                        related_chunks.append({
                            'id': neighbor_id,
                            'chunk_uri': engine.generate_chunk_uri(dict(chunk_meta)),
                            'text': chunk_doc,
                            'meta': chunk_meta,
                            'relationship': neighbor['relationship'],
                            'source_chunk': source_chunk_id,
                            'importance_score': importance,
                            'retrieval_method': 'graph_traversal'
                        })
        
        # Sort by importance and limit results
        related_chunks.sort(key=lambda x: x['importance_score'], reverse=True)
        return related_chunks[:max_results]
        
    except Exception as e:
        logger.error(f"Error getting related chunks for query '{query}': {e}")
        return []

@mcp.tool()
async def explore_chunk_context(
    chunk_id: str,
    context_window: int = 2
) -> Dict[str, Any]:
    """Get surrounding context for a specific chunk.
    
    Returns the chunk along with its sequential neighbors and hierarchical context,
    useful for understanding the full context around a specific piece of content.
    """
    state = await get_app_state()
    try:
        col = state.unified_store._collection()
        
        # Get the target chunk
        chunk_results = col.get(
            where={"chunk_id": {"$eq": chunk_id}},
            include=['metadatas', 'documents']
        )
        
        chunk_meta, chunk_doc = safe_get_chroma_results(chunk_results)
        if not chunk_meta:
            raise ValueError(f"Chunk not found: {chunk_id}")
        
        # Get sequential context (previous and next chunks)
        sequential_context = []
        neighbors = state.unified_store.get_chunk_neighbors(
            chunk_id, 
            include_sequential=True, 
            include_hierarchical=False
        )
        
        for neighbor in neighbors:
            if neighbor['relationship'] in ['sequential_prev', 'sequential_next']:
                neighbor_results = col.get(
                    where={"chunk_id": {"$eq": neighbor['chunk_id']}},
                    include=['metadatas', 'documents']
                )
                
                neighbor_meta, neighbor_doc = safe_get_chroma_results(neighbor_results)
                if neighbor_meta:
                    engine = await get_smart_search_engine()
                    sequential_context.append({
                        'chunk_id': neighbor['chunk_id'],
                        'position': neighbor['relationship'],
                        'content': neighbor_doc,
                        'header_text': neighbor_meta.get('header_text', ''),
                        'chunk_type': neighbor_meta.get('chunk_type', ''),
                        'chunk_uri': engine.generate_chunk_uri(dict(neighbor_meta))
                    })
        
        # Get hierarchical context (parent and children)
        hierarchical_context = []
        hierarchy_neighbors = state.unified_store.get_chunk_neighbors(
            chunk_id,
            include_sequential=False,
            include_hierarchical=True
        )
        
        for neighbor in hierarchy_neighbors:
            if neighbor['relationship'] in ['parent', 'child']:
                neighbor_results = col.get(
                    where={"chunk_id": {"$eq": neighbor['chunk_id']}},
                    include=['metadatas', 'documents']
                )
                
                neighbor_meta, neighbor_doc = safe_get_chroma_results(neighbor_results)
                if neighbor_meta:
                    engine = await get_smart_search_engine()
                    hierarchical_context.append({
                        'chunk_id': neighbor['chunk_id'],
                        'relationship': neighbor['relationship'],
                        'content': neighbor_doc,
                        'header_text': neighbor_meta.get('header_text', ''),
                        'header_level': neighbor_meta.get('header_level', 0),
                        'chunk_type': neighbor_meta.get('chunk_type', ''),
                        'chunk_uri': engine.generate_chunk_uri(dict(neighbor_meta))
                    })
        
        # Build full context including the target chunk
        parent_headers = chunk_meta.get('parent_headers', '')
        if parent_headers:
            header_hierarchy = parent_headers.split(',') if isinstance(parent_headers, str) else []
        else:
            header_hierarchy = []
        
        # Get engine for generating URI
        engine = await get_smart_search_engine()
        
        return {
            'target_chunk': {
                'chunk_id': chunk_id,
                'content': chunk_doc,
                'header_text': chunk_meta.get('header_text', ''),
                'header_level': chunk_meta.get('header_level', 0),
                'chunk_type': chunk_meta.get('chunk_type', ''),
                'importance_score': chunk_meta.get('importance_score', 0.5),
                'chunk_uri': engine.generate_chunk_uri(dict(chunk_meta)),
                'note_title': chunk_meta.get('title', ''),
                'path': chunk_meta.get('path', '')
            },
            'sequential_context': sorted(sequential_context, 
                                       key=lambda x: 0 if x['position'] == 'sequential_prev' else 1),
            'hierarchical_context': hierarchical_context,
            'header_hierarchy': header_hierarchy,
            'note_context': {
                'note_id': chunk_meta.get('note_id', ''),
                'title': chunk_meta.get('title', ''),
                'tags': str(chunk_meta.get('tags', '')).split(',') if chunk_meta.get('tags') else [],
                'links': str(chunk_meta.get('links_to', '')).split(',') if chunk_meta.get('links_to') else []
            }
        }
        
    except Exception as e:
        logger.error(f"Error exploring context for chunk {chunk_id}: {e}")
        raise

def _get_relationship_description(rel_type: str) -> str:
    """Get human-readable description of relationship type."""
    descriptions = {
        'sequential_next': 'Chunks that come immediately after in the document',
        'sequential_prev': 'Chunks that come immediately before in the document',
        'parent': 'Parent sections or headers that contain this chunk',
        'child': 'Subsections or content under this chunk',
        'sibling': 'Chunks at the same hierarchical level',
        'content_link': 'Chunks from notes that are linked to from this content'
    }
    return descriptions.get(rel_type, f'Related through {rel_type}')

class ReindexResult(BaseModel):
    """Result of reindexing operation."""
    operation: str
    notes_indexed: int
    success: bool
    message: str

@mcp.tool()
async def reindex_vault(
    target: str = "all",
    full_reindex: bool = False
) -> ReindexResult:
    state = await get_app_state()
    """
    Reindex the vault with unified store.
    
    Args:
        target: What to reindex - "all" (unified store supports all data)
        full_reindex: If True, completely rebuild the database (default: False)
    
    Returns:
        ReindexResult with operation details
    """
    try:
        # Use unified store's reindex method
        notes_count = state.unified_store.reindex(
            vaults=settings.vaults,
            full_reindex=full_reindex
        )
        
        return ReindexResult(
            operation=f"reindex_{target}",
            notes_indexed=notes_count,
            success=True,
            message=f"Successfully reindexed {notes_count} chunks from vault"
        )
    
    except Exception as e:
        return ReindexResult(
            operation=f"reindex_{target}",
            notes_indexed=0,
            success=False,
            message=f"Reindex failed: {str(e)}"
        )

class EnrichmentResult(BaseModel):
    """Result of note enrichment."""
    processed_notes: int
    successful: int
    failed: int
    para_distribution: Dict[str, int]
    message: str

@mcp.tool()
async def enrich_notes(
    note_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    dry_run: bool = False
) -> EnrichmentResult:
    """
    Enrich notes with PARA taxonomy and semantic relationships.
    
    Args:
        note_paths: Specific note paths to enrich. If None, enriches all notes.
        limit: Maximum number of notes to process (if note_paths is None)
        dry_run: If True, analyze but don't save changes (default: False)
    
    Returns:
        EnrichmentResult with enrichment statistics
    """
    try:
        # Import enrichment module dynamically
        import importlib.util
        script_path = Path(__file__).parent.parent / "scripts" / "enrich_para_taxonomy.py"
        spec = importlib.util.spec_from_file_location("enrich_para_taxonomy", script_path)
        ParaTaxonomyEnricher = None
        if spec and spec.loader:
            enrich_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(enrich_module)
            ParaTaxonomyEnricher = enrich_module.ParaTaxonomyEnricher
        
        # Initialize enricher
        if ParaTaxonomyEnricher is None:
            raise ImportError("Could not import ParaTaxonomyEnricher")
        enricher = ParaTaxonomyEnricher()
        
        # Determine which notes to process
        if note_paths:
            # Filter out protected test corpus paths
            paths_to_process: list[str] = [
                str(p) for p in (Path(p) for p in note_paths)
                if not is_protected_test_content(Path(p))
            ]
        else:
            # Get all notes from vault
            from .fs_indexer import discover_files
            all_paths = [
                p for p in discover_files(settings.vaults, settings.supported_extensions)
                if not is_protected_test_content(p)
            ]
            
            # Apply limit if specified
            if limit:
                paths_to_process = [str(p) for p in all_paths[:limit]]
            else:
                paths_to_process = [str(p) for p in all_paths]
        
        # Process notes
        processed = 0
        successful = 0
        failed = 0
        para_distribution = {}
        
        for note_path in paths_to_process:
            try:
                result = enricher.enrich_note_properties(note_path, dry_run=dry_run)
                if result:
                    processed += 1
                    successful += 1
                    
                    # Track PARA distribution
                    para_type = result.get('para_type', 'unknown')
                    para_distribution[para_type] = para_distribution.get(para_type, 0) + 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        
        mode = "Dry run - no changes applied" if dry_run else "Changes applied"
        
        return EnrichmentResult(
            processed_notes=processed,
            successful=successful,
            failed=failed,
            para_distribution=para_distribution,
            message=f"Enrichment complete. {mode}. Processed {processed} notes: {successful} successful, {failed} failed."
        )
    
    except Exception as e:
        return EnrichmentResult(
            processed_notes=0,
            successful=0,
            failed=0,
            para_distribution={},
            message=f"Enrichment failed: {str(e)}"
        )

# ============= Base File Tools =============

@mcp.tool()
async def list_bases() -> List[BaseInfo]:
    """List all .base files in the vault with their metadata.
    
    Returns:
        List of BaseInfo objects containing base file information
    """
    state = await get_app_state()
    try:
        bases = state.base_manager.discover_bases()
        logger.info(f"Found {len(bases)} base files in vault")
        return bases
    except Exception as e:
        logger.error(f"Failed to list bases: {e}")
        return []

@mcp.tool()
async def read_base(base_id: str) -> Optional[Dict[str, Any]]:
    """Read and parse a .base file by its ID.
    
    Args:
        base_id: The base file ID to read
        
    Returns:
        Parsed base file configuration as dictionary, or None if not found
    """
    state = await get_app_state()
    try:
        base = state.base_manager.get_base(base_id)
        if base:
            return base.model_dump(by_alias=True, exclude_none=True)
        return None
    except Exception as e:
        logger.error(f"Failed to read base {base_id}: {e}")
        return None

@mcp.tool()
async def validate_base(content: str, format: Optional[Literal["json", "yaml"]] = None) -> Dict[str, Any]:
    """Validate .base file content syntax and structure.
    
    Args:
        content: The .base file content to validate
        format: Optional format hint (json or yaml)
        
    Returns:
        Dictionary with 'valid' boolean and optional 'error' message
    """
    try:
        is_valid, error = BaseParser.validate(content, format)
        return {"valid": is_valid, "error": error}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@mcp.tool()
async def execute_base_query(
    base_id: str, 
    view_id: Optional[str] = None,
    limit: Optional[int] = None
) -> Optional[BaseQueryResult]:
    """Execute a base query and return matching notes.
    
    Args:
        base_id: The base file ID to execute
        view_id: Optional specific view to use
        limit: Optional limit on number of results
        
    Returns:
        Query results with matching notes and metadata
    """
    state = await get_app_state()
    try:
        result = state.base_manager.execute_query(base_id, view_id)
        
        # Apply limit if specified
        if limit and result.results:
            result.results = result.results[:limit]
            result.filtered_count = min(result.filtered_count, limit)
        
        return result
    except Exception as e:
        logger.error(f"Failed to execute base query {base_id}: {e}")
        return None

@mcp.tool()
async def get_base_view(base_id: str, view_id: str) -> Optional[BaseViewData]:
    """Get formatted data for a specific base view.
    
    Args:
        base_id: The base file ID
        view_id: The view ID to format
        
    Returns:
        Formatted view data for display
    """
    state = await get_app_state()
    try:
        # First execute the query
        query_result = state.base_manager.execute_query(base_id, view_id)
        if not query_result:
            return None
        
        # Format for the specific view
        view_data = state.base_manager.format_view_data(query_result, base_id, view_id)
        return view_data
    except Exception as e:
        logger.error(f"Failed to get base view {base_id}/{view_id}: {e}")
        return None

@mcp.tool()
async def create_base(
    name: str,
    description: Optional[str] = None,
    folders: Optional[List[str]] = None,
    filters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Create a new .base file with basic configuration.
    
    Args:
        name: Name for the new base
        description: Optional description
        folders: Folders to include (defaults to root)
        filters: Optional filter conditions
        
    Returns:
        Dictionary with 'success' boolean, 'base_id', 'path', and optional 'error'
    """
    try:
        from base_parser import BaseFile, BaseSource, BaseView, BaseColumn, ViewType
        
        # Generate base ID from name
        base_id = name.lower().replace(' ', '-').replace('_', '-')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '-')
        logger.info(f"Creating base with ID: {base_id}")
        
        # Create base structure
        base = BaseFile(
            id=base_id,
            name=name,
            version=1,
            description=description,
            source=BaseSource(
                folders=folders or ["/"],
                includeSubfolders=True,
                filters=filters or []
            ),
            views=[
                BaseView(
                    id="table-main",
                    name="Table",
                    type=ViewType.TABLE,
                    columns=[
                        BaseColumn(id="title", header="Title", source="title", linkTo="file"),
                        BaseColumn(id="path", header="Path", source="path"),
                        BaseColumn(id="tags", header="Tags", source="tags"),
                        BaseColumn(id="modified", header="Modified", source="file.mtime")
                    ]
                )
            ]
        )
        
        # Save to vault
        vault_path = settings.vaults[0] if settings.vaults else Path.cwd()
        base_path = vault_path / f"{base_id}.base"
        logger.info(f"Saving base file to: {base_path}")
        
        # Check if file already exists
        if base_path.exists():
            logger.warning(f"Base file already exists at: {base_path}")
        
        # Write as JSON
        base_json = BaseParser.to_json(base)
        base_path.write_text(base_json, encoding='utf-8')
        
        # Verify the file was written
        if not base_path.exists():
            logger.error(f"Failed to create file at: {base_path}")
            return {
                "success": False,
                "error": f"File was not created at {base_path}"
            }
        
        file_size = base_path.stat().st_size
        logger.info(f"Successfully created base file: {base_path} ({file_size} bytes)")
        
        # Return both relative and absolute paths for clarity
        return {
            "success": True,
            "base_id": base_id,
            "path": str(base_path.relative_to(vault_path)),
            "absolute_path": str(base_path),
            "file_size": file_size
        }
        
    except Exception as e:
        logger.error(f"Failed to create base: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@mcp.tool()
async def list_base_files() -> Dict[str, Any]:
    """List all .base files in the vault.
    
    Returns:
        Dictionary with 'success' boolean, 'base_files' list, and optional 'error'
    """
    try:
        vault_path = settings.vaults[0] if settings.vaults else Path.cwd()
        base_files = list(vault_path.glob("*.base"))
        
        files_info = []
        for base_file in base_files:
            try:
                # Get file info
                stat = base_file.stat()
                files_info.append({
                    "name": base_file.name,
                    "path": str(base_file.relative_to(vault_path)),
                    "absolute_path": str(base_file),
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
            except Exception as e:
                logger.warning(f"Could not get info for {base_file}: {e}")
        
        logger.info(f"Found {len(files_info)} base files in {vault_path}")
        
        return {
            "success": True,
            "vault_path": str(vault_path),
            "base_files": files_info,
            "count": len(files_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to list base files: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@mcp.tool()
async def check_base_exists(base_id: str) -> Dict[str, Any]:
    """Check if a base file exists.
    
    Args:
        base_id: The base file ID to check
        
    Returns:
        Dictionary with 'exists' boolean, file info if exists, and optional 'error'
    """
    try:
        vault_path = settings.vaults[0] if settings.vaults else Path.cwd()
        base_path = vault_path / f"{base_id}.base"
        
        if base_path.exists():
            stat = base_path.stat()
            return {
                "exists": True,
                "path": str(base_path.relative_to(vault_path)),
                "absolute_path": str(base_path),
                "size": stat.st_size,
                "modified": stat.st_mtime
            }
        else:
            # Check if there's a file with different extension or case
            similar_files = list(vault_path.glob(f"{base_id}.*"))
            similar_files.extend(vault_path.glob(f"{base_id.upper()}.*"))
            similar_files.extend(vault_path.glob(f"{base_id.lower()}.*"))
            
            return {
                "exists": False,
                "expected_path": str(base_path),
                "similar_files": [str(f.relative_to(vault_path)) for f in similar_files]
            }
            
    except Exception as e:
        logger.error(f"Failed to check base exists: {e}", exc_info=True)
        return {"exists": False, "error": str(e)}

@mcp.tool()
async def update_base(
    base_id: str,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Update an existing .base file configuration.
    
    Args:
        base_id: The base file ID to update
        updates: Dictionary of updates to apply
        
    Returns:
        Dictionary with 'success' boolean and optional 'error'
    """
    state = await get_app_state()
    try:
        # Get existing base
        base = state.base_manager.get_base(base_id)
        if not base:
            return {"success": False, "error": f"Base not found: {base_id}"}
        
        # Apply updates
        base_dict = base.model_dump(by_alias=True)
        
        # Deep merge updates
        def deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(base_dict, updates)
        
        # Validate updated structure
        updated_base = BaseFile.model_validate(base_dict)
        
        # Save back to file
        base_path = state.base_manager.get_base_path(base_id)
        if base_path:
            base_json = BaseParser.to_json(updated_base)
            base_path.write_text(base_json, encoding='utf-8')
            return {"success": True}
        else:
            return {"success": False, "error": "Could not locate base file path"}
        
    except Exception as e:
        logger.error(f"Failed to update base {base_id}: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def base_from_graph(
    note_ids: List[str],
    name: str,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new .base file from a selection of notes (graph-friendly).
    
    This tool enables creating bases from graph selections, making it easy
    to save curated collections of related notes discovered through graph traversal.
    
    Args:
        note_ids: List of note IDs to include
        name: Name for the new base
        description: Optional description
        
    Returns:
        Dictionary with 'success', 'base_id', 'path', and optional 'error'
    """
    state = await get_app_state()
    try:
        # Create base from selected notes
        base = state.base_manager.create_base_from_graph(note_ids, name, description)
        
        # Save to vault
        vault_path = settings.vaults[0] if settings.vaults else Path.cwd()
        base_path = vault_path / f"{base.id}.base"
        
        # Write as JSON
        base_json = BaseParser.to_json(base)
        base_path.write_text(base_json, encoding='utf-8')
        
        return {
            "success": True,
            "base_id": base.id,
            "path": str(base_path.relative_to(vault_path)),
            "note_count": len(note_ids)
        }
        
    except Exception as e:
        logger.error(f"Failed to create base from graph: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def enrich_base_with_graph(base_id: str) -> Dict[str, Any]:
    """Enrich a base with graph relationship data as computed fields.
    
    Adds computed fields for graph metrics like link count, centrality,
    and cluster information to enhance base views with graph insights.
    
    Args:
        base_id: The base file ID to enrich
        
    Returns:
        Dictionary with 'success' and enrichment details
    """
    state = await get_app_state()
    try:
        # Get the base
        base = state.base_manager.get_base(base_id)
        if not base:
            return {"success": False, "error": f"Base not found: {base_id}"}
        
        from base_parser import ComputedField, ComputedType
        
        # Add graph-related computed fields
        new_computed = [
            ComputedField(
                id="link-count",
                expr="len(links_to)",
                type=ComputedType.NUMBER
            ),
            ComputedField(
                id="backlink-count", 
                expr="len(backlinks_from)",
                type=ComputedType.NUMBER
            ),
            ComputedField(
                id="connectivity-score",
                expr="len(links_to) + len(backlinks_from)",
                type=ComputedType.NUMBER
            ),
            ComputedField(
                id="has-tags",
                expr="len(tags) > 0",
                type=ComputedType.BOOLEAN
            )
        ]
        
        # Add to base if not already present
        existing_ids = {cf.id for cf in base.computed}
        for cf in new_computed:
            if cf.id not in existing_ids:
                base.computed.append(cf)
        
        # Save updated base
        base_path = state.base_manager.get_base_path(base_id)
        if base_path:
            base_json = BaseParser.to_json(base)
            base_path.write_text(base_json, encoding='utf-8')
            
            return {
                "success": True,
                "added_fields": [cf.id for cf in new_computed if cf.id not in existing_ids]
            }
        else:
            return {"success": False, "error": "Could not locate base file path"}
            
    except Exception as e:
        logger.error(f"Failed to enrich base {base_id}: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Check the health status of the MCP server and its components.
    
    Returns comprehensive health information including:
    - Overall status
    - Component health checks
    - Cache statistics
    - System metrics
    """
    state = await get_app_state()
    try:
        # Run all health checks
        health_results = await state.health_check.run_checks()
        
        # Get cache statistics
        cache_stats = await cache_manager.get_all_stats()
        
        # Get rate limiter status
        rate_limiter_status = {
            "rate": state.rate_limiter.rate,
            "burst": state.rate_limiter.burst,
            "available_tokens": state.rate_limiter.tokens
        }
        
        # Get ChromaDB metrics
        try:
            collection_count = state.unified_store._collection().count()
            chromadb_status = "healthy"
        except Exception as e:
            collection_count = 0
            chromadb_status = f"error: {str(e)}"
        
        return {
            "status": health_results["status"],
            "timestamp": health_results["timestamp"],
            "components": health_results["checks"],
            "metrics": {
                "chromadb": {
                    "status": chromadb_status,
                    "document_count": collection_count,
                    "collection": settings.collection
                },
                "rate_limiter": rate_limiter_status,
                "optimization": {
                    "enabled": settings.dspy_optimize_enabled,
                    "available": DSPY_OPTIMIZATION_AVAILABLE
                }
            },
            "caches": cache_stats,
            "configuration": {
                "vault_count": len(settings.vaults),
                "chunk_strategy": settings.chunk_strategy,
                "embedding_model": settings.embedding_model,
                "gemini_model": settings.gemini_model
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def get_cache_stats() -> Dict[str, Any]:
    """Get detailed cache statistics for monitoring and optimization."""
    try:
        stats = await cache_manager.get_all_stats()
        
        # Calculate overall metrics
        total_hits = sum(s.get("hits", 0) for s in stats.values())
        total_misses = sum(s.get("misses", 0) for s in stats.values())
        total_evictions = sum(s.get("evictions", 0) for s in stats.values())
        total_size = sum(s.get("size", 0) for s in stats.values())
        
        overall_hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0
        
        return {
            "overall": {
                "total_caches": len(stats),
                "total_items": total_size,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_evictions": total_evictions,
                "overall_hit_rate": f"{overall_hit_rate:.2f}%"
            },
            "caches": stats,
            "recommendations": _get_cache_recommendations(stats)
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {"error": str(e)}

def _get_cache_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate cache optimization recommendations based on statistics."""
    recommendations = []
    
    for cache_name, cache_stats in stats.items():
        # Check hit rate
        hit_rate_str = cache_stats.get("hit_rate", "0%")
        hit_rate = float(hit_rate_str.rstrip('%'))
        
        if hit_rate < 20:
            recommendations.append(f"Cache '{cache_name}' has low hit rate ({hit_rate_str}). Consider adjusting TTL or reviewing usage patterns.")
        
        # Check eviction rate
        evictions = cache_stats.get("evictions", 0)
        size = cache_stats.get("size", 0)
        max_size = cache_stats.get("max_size", 1000)
        
        if evictions > max_size * 2:
            recommendations.append(f"Cache '{cache_name}' has high eviction count ({evictions}). Consider increasing max_size.")
        
        # Check utilization
        utilization = (size / max_size * 100) if max_size > 0 else 0
        if utilization < 10:
            recommendations.append(f"Cache '{cache_name}' is underutilized ({utilization:.1f}%). Consider reducing max_size.")
    
    if not recommendations:
        recommendations.append("All caches operating within normal parameters.")
    
    return recommendations

@mcp.tool()
async def clear_caches(cache_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Clear specified caches or all caches if no names provided.
    
    Args:
        cache_names: Optional list of cache names to clear. If None, clears all caches.
    
    Returns:
        Status of cache clearing operation.
    """
    try:
        if cache_names:
            cleared = []
            for name in cache_names:
                cache = cache_manager.get_cache(name)
                if cache:
                    await cache.clear()
                    cleared.append(name)
                else:
                    logger.warning(f"Cache '{name}' not found")
            
            return {
                "success": True,
                "cleared": cleared,
                "message": f"Cleared {len(cleared)} cache(s)"
            }
        else:
            await cache_manager.clear_all()
            return {
                "success": True,
                "message": "All caches cleared"
            }
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def force_dspy_optimization() -> Dict[str, Any]:
    """Force immediate DSPy optimization of RAG programs."""
    state = await get_app_state()
    
    if not settings.dspy_optimize_enabled:
        return {
            "success": False,
            "message": "DSPy optimization is disabled",
            "status": "disabled"
        }
    
    if not DSPY_OPTIMIZATION_AVAILABLE:
        return {
            "success": False,
            "message": "DSPy optimization modules not available",
            "status": "unavailable"
        }
    
    try:
        # Check if searcher has optimization capabilities
        if hasattr(state.searcher, 'force_optimization'):
            logger.info("Forcing DSPy optimization via MCP tool")
            result = state.searcher.force_optimization()
            return {
                "success": True,
                "message": "DSPy optimization triggered successfully",
                "optimization_result": result,
                "status": "completed"
            }
        else:
            return {
                "success": False,
                "message": "Searcher does not support optimization",
                "status": "not_supported"
            }
            
    except Exception as e:
        logger.error(f"Failed to force DSPy optimization: {e}")
        return {
            "success": False,
            "message": f"Optimization failed: {str(e)}",
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
async def get_dspy_optimization_status() -> Dict[str, Any]:
    """Get current DSPy optimization status and metrics."""
    state = await get_app_state()
    
    if not settings.dspy_optimize_enabled:
        return {
            "optimization_enabled": False,
            "status": "disabled",
            "message": "DSPy optimization is disabled in settings"
        }
    
    if not DSPY_OPTIMIZATION_AVAILABLE:
        return {
            "optimization_enabled": False,
            "status": "unavailable",
            "message": "DSPy optimization modules not available"
        }
    
    try:
        # Get optimization status from searcher
        if hasattr(state.searcher, 'get_optimization_status'):
            status = state.searcher.get_optimization_status()
            status["mcp_tool_available"] = True
            return status
        else:
            return {
                "optimization_enabled": False,
                "status": "not_supported",
                "message": "Searcher does not support optimization status"
            }
            
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        return {
            "optimization_enabled": False,
            "status": "error",
            "message": f"Status check failed: {str(e)}",
            "error": str(e)
        }

def run_stdio():
    """Run MCP server via stdio for Claude Desktop integration."""
    mcp.run()

def run_http(host: Optional[str] = None, port: Optional[int] = None):
    """Run MCP server via HTTP for Cursor and other HTTP clients."""
    import uvicorn
    from .config import settings
    
    # Use provided values or fall back to settings
    host = host or settings.mcp_host
    port = port or settings.mcp_port
    
    logger.info("Starting Graph RAG MCP Server (HTTP) on %s:%s", host, port)
    logger.info("Vault paths: %s", [str(p) for p in settings.vaults])
    logger.info("Unified Store (ChromaDB): %s", settings.chroma_dir)
    
    # Run FastMCP with HTTP transport
    uvicorn.run(
        "src.mcp_server:mcp.get_app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run_stdio()
