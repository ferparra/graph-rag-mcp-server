"""Manager for Obsidian .base files with graph integration."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, cast
from datetime import datetime
from collections import defaultdict

from pydantic import BaseModel, Field

# Support both package and module execution contexts
try:
    from base_parser import BaseFile, BaseParser, FilterOperator, ViewType
    from unified_store import UnifiedStore
    from config import settings
except ImportError:
    from .base_parser import BaseFile, BaseParser, FilterOperator, ViewType
    from .unified_store import UnifiedStore
    from .config import settings

logger = logging.getLogger(__name__)


class BaseQueryResult(BaseModel):
    """Result from executing a base query."""
    base_id: str
    view_id: Optional[str] = None
    total_count: int
    filtered_count: int
    results: List[Dict[str, Any]]
    groups: Optional[Dict[str, List[Dict[str, Any]]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseViewData(BaseModel):
    """Formatted view data for display."""
    base_id: str
    view_id: str
    view_type: str
    columns: Optional[List[Dict[str, Any]]] = None
    rows: Optional[List[Dict[str, Any]]] = None
    cards: Optional[List[Dict[str, Any]]] = None
    groups: Optional[Dict[str, List[Any]]] = None
    total_count: int
    page_size: Optional[int] = None


class BaseInfo(BaseModel):
    """Information about a .base file."""
    id: str
    name: str
    description: Optional[str]
    path: str
    version: int
    view_count: int
    source_folders: List[str]
    filter_count: int
    computed_count: int
    last_modified: Optional[datetime] = None


class BaseManager:
    """Manages .base files and executes queries using the unified store."""
    
    def __init__(self, unified_store: UnifiedStore, vault_path: Optional[Path] = None):
        """Initialize the base manager.
        
        Args:
            unified_store: The unified ChromaDB store
            vault_path: Optional vault path for discovering .base files
        """
        self.unified_store = unified_store
        self.vault_path = vault_path or settings.vaults[0] if settings.vaults else Path.cwd()
        self._base_cache: Dict[str, BaseFile] = {}
        self._expression_cache: Dict[str, Any] = {}
    
    def discover_bases(self) -> List[BaseInfo]:
        """Discover all .base files in the vault.
        
        Returns:
            List of BaseInfo objects
        """
        bases = []
        base_pattern = "**/*.base"
        
        for base_path in self.vault_path.glob(base_pattern):
            try:
                base = self._load_base(base_path)
                if base:
                    bases.append(BaseInfo(
                        id=base.id,
                        name=base.name,
                        description=base.description,
                        path=str(base_path.relative_to(self.vault_path)),
                        version=base.version,
                        view_count=len(base.views),
                        source_folders=base.source.folders,
                        filter_count=len(base.source.filters),
                        computed_count=len(base.computed),
                        last_modified=datetime.fromtimestamp(base_path.stat().st_mtime)
                    ))
            except Exception as e:
                logger.warning(f"Failed to load base file {base_path}: {e}")
        
        return sorted(bases, key=lambda b: b.name)
    
    def _load_base(self, path: Path) -> Optional[BaseFile]:
        """Load and cache a base file.
        
        Args:
            path: Path to the .base file
            
        Returns:
            Parsed BaseFile or None if failed
        """
        try:
            # Check cache
            cache_key = str(path.absolute())
            if cache_key in self._base_cache:
                # Check if file has been modified
                cached_base = self._base_cache[cache_key]
                if hasattr(cached_base, '_mtime'):
                    current_mtime = path.stat().st_mtime
                    if cached_base._mtime == current_mtime:  # type: ignore
                        return cached_base
            
            # Parse file
            base = BaseParser.parse_file(path)
            base._mtime = path.stat().st_mtime  # type: ignore
            base._path = path  # type: ignore
            
            # Cache it
            self._base_cache[cache_key] = base
            return base
            
        except Exception as e:
            logger.error(f"Failed to load base file {path}: {e}")
            return None
    
    def get_base(self, base_id: str) -> Optional[BaseFile]:
        """Get a base file by ID.
        
        Args:
            base_id: The base ID to find
            
        Returns:
            BaseFile if found, None otherwise
        """
        # First check cache
        for cached_base in self._base_cache.values():
            if cached_base.id == base_id:
                return cached_base
        
        # Search vault for matching base
        for base_path in self.vault_path.glob("**/*.base"):
            base = self._load_base(base_path)
            if base and base.id == base_id:
                return base
        
        return None
    
    def _convert_filter_to_chroma(self, filter_: Any) -> Optional[Dict[str, Any]]:
        """Convert a base filter to ChromaDB where clause.
        
        Args:
            filter_: BaseFilter object
            
        Returns:
            ChromaDB where clause or None
        """
        prop = filter_.property
        op = filter_.op
        value = filter_.value
        
        # Map operators to ChromaDB operators
        if op == FilterOperator.EQ:
            return {prop: {"$eq": value}}
        elif op == FilterOperator.NEQ:
            return {prop: {"$ne": value}}
        elif op == FilterOperator.GT:
            return {prop: {"$gt": value}}
        elif op == FilterOperator.GTE:
            return {prop: {"$gte": value}}
        elif op == FilterOperator.LT:
            return {prop: {"$lt": value}}
        elif op == FilterOperator.LTE:
            return {prop: {"$lte": value}}
        elif op == FilterOperator.IN:
            return {prop: {"$in": value if isinstance(value, list) else [value]}}
        elif op == FilterOperator.NIN:
            return {prop: {"$nin": value if isinstance(value, list) else [value]}}
        elif op == FilterOperator.CONTAINS:
            # For contains, we'll need to handle this specially
            # ChromaDB doesn't have a direct contains operator for metadata
            # We'll filter in post-processing
            return None
        elif op == FilterOperator.EXISTS:
            # Check if property exists (not None)
            return {prop: {"$ne": None}}
        elif op == FilterOperator.NEXISTS:
            # Check if property doesn't exist (is None)
            return {prop: {"$eq": None}}
        elif op == FilterOperator.REGEX:
            # Regex also needs post-processing
            return None
        
        return None
    
    def _apply_post_filters(self, results: List[Dict[str, Any]], filters: List[Any]) -> List[Dict[str, Any]]:
        """Apply filters that can't be handled by ChromaDB directly.
        
        Args:
            results: Initial results from ChromaDB
            filters: List of BaseFilter objects
            
        Returns:
            Filtered results
        """
        import re
        
        filtered = []
        for result in results:
            include = True
            
            for filter_ in filters:
                prop = filter_.property
                op = filter_.op
                value = filter_.value
                
                # Get property value from result
                if prop.startswith('file.'):
                    # Handle file properties
                    prop_value = result.get(prop)
                else:
                    # Regular property
                    prop_value = result.get(prop)
                
                # Apply filter based on operator
                if op == FilterOperator.CONTAINS:
                    if prop_value is None or str(value).lower() not in str(prop_value).lower():
                        include = False
                        break
                elif op == FilterOperator.NCONTAINS:
                    if prop_value is not None and str(value).lower() in str(prop_value).lower():
                        include = False
                        break
                elif op == FilterOperator.REGEX:
                    if prop_value is None:
                        include = False
                        break
                    try:
                        if not re.search(value, str(prop_value)):
                            include = False
                            break
                    except re.error:
                        logger.warning(f"Invalid regex pattern: {value}")
                        include = False
                        break
            
            if include:
                filtered.append(result)
        
        return filtered
    
    def execute_query(self, base_id: str, view_id: Optional[str] = None, 
                     additional_filters: Optional[List[Dict[str, Any]]] = None) -> BaseQueryResult:
        """Execute a base query and return results.
        
        Args:
            base_id: The base ID to execute
            view_id: Optional specific view to use
            additional_filters: Additional filters to apply
            
        Returns:
            Query results
        """
        base = self.get_base(base_id)
        if not base:
            raise ValueError(f"Base not found: {base_id}")
        
        # Find the view
        view = None
        if view_id:
            view = next((v for v in base.views if v.id == view_id), None)
            if not view:
                raise ValueError(f"View not found: {view_id}")
        else:
            # Use first view as default
            view = base.views[0] if base.views else None
        
        # Build ChromaDB query
        where_clauses = []
        post_filters = []
        
        # Convert base filters
        for filter_ in base.source.filters:
            chroma_filter = self._convert_filter_to_chroma(filter_)
            if chroma_filter:
                where_clauses.append(chroma_filter)
            else:
                # Need post-processing for this filter
                post_filters.append(filter_)
        
        # Add folder filters
        folder_filters = []
        for folder in base.source.folders:
            # Convert folder to absolute path pattern
            vault_folder = self.vault_path / folder
            if base.source.includeSubfolders:
                pattern = f"{vault_folder}/**"
            else:
                pattern = str(vault_folder)
            folder_filters.append({"path": {"$contains": pattern}})
        
        # Combine filters
        where = {}
        if where_clauses:
            if len(where_clauses) == 1:
                where = where_clauses[0]
            else:
                where = {"$and": where_clauses}
        
        # Query unified store
        try:
            collection = self.unified_store._collection()
            
            # Get all documents matching the filters
            results = collection.get(
                where=cast(Any, where) if where else None,
                include=['metadatas', 'documents']
            )
            
            # Process results
            processed_results = []
            metadatas = results.get('metadatas', []) or []
            documents = results.get('documents', []) or []
            
            # Ensure metadatas and documents are lists
            if not isinstance(metadatas, list):
                metadatas = []
            if not isinstance(documents, list):
                documents = []
            
            for i, metadata in enumerate(metadatas):
                if metadata and isinstance(metadata, dict):
                    result = dict(metadata)  # Create a copy using dict constructor
                    result['_content'] = documents[i] if i < len(documents) else ""
                    processed_results.append(result)
            
            # Apply post-filters
            if post_filters:
                processed_results = self._apply_post_filters(processed_results, post_filters)
            
            # Apply computed fields if any
            if base.computed:
                processed_results = self._apply_computed_fields(processed_results, base)
            
            # Sort results if view has sort config
            if view and view.sort:
                processed_results = self._sort_results(processed_results, view.sort)
            
            # Group results if view has group config
            groups = None
            if view and view.group:
                groups = self._group_results(processed_results, view.group)
            
            return BaseQueryResult(
                base_id=base_id,
                view_id=view_id if view_id else (view.id if view else None),
                total_count=len(metadatas) if metadatas else 0,
                filtered_count=len(processed_results),
                results=processed_results,
                groups=groups,
                metadata={
                    "base_name": base.name,
                    "view_name": view.name if view else None,
                    "view_type": view.type.value if view else None
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to execute query for base {base_id}: {e}")
            raise
    
    def _apply_computed_fields(self, results: List[Dict[str, Any]], base: BaseFile) -> List[Dict[str, Any]]:
        """Apply computed fields to results.
        
        Args:
            results: Query results
            base: Base file with computed field definitions
            
        Returns:
            Results with computed fields added
        """
        # This will be implemented with the expression evaluator
        # For now, just return results as-is
        logger.info(f"Computed fields not yet implemented, skipping {len(base.computed)} fields")
        return results
    
    def _sort_results(self, results: List[Dict[str, Any]], sort_configs: List[Any]) -> List[Dict[str, Any]]:
        """Sort results according to sort configuration.
        
        Args:
            results: Results to sort
            sort_configs: List of SortConfig objects
            
        Returns:
            Sorted results
        """
        if not sort_configs:
            return results
        
        def get_sort_value(item: Dict[str, Any], key: str) -> Any:
            """Get value for sorting, handling special cases."""
            if key.startswith('@'):
                # Computed field reference
                return item.get(key[1:])
            elif key.startswith('file.'):
                # File property
                return item.get(key)
            else:
                # Regular property
                return item.get(key)
        
        # Apply sorts in reverse order (last sort is primary)
        for sort_cfg in reversed(sort_configs):
            reverse = sort_cfg.dir.value == 'desc'
            
            # Handle null positioning
            def sort_key(item):
                value = get_sort_value(item, sort_cfg.by)
                if value is None:
                    # Position nulls according to config
                    if sort_cfg.nulls.value == 'first':
                        return (0, None) if not reverse else (1, None)
                    else:  # 'last'
                        return (1, None) if not reverse else (0, None)
                else:
                    return (0 if sort_cfg.nulls.value == 'first' else 1, value)
            
            results = sorted(results, key=sort_key, reverse=reverse)
        
        return results
    
    def _group_results(self, results: List[Dict[str, Any]], group_config: Any) -> Dict[str, List[Dict[str, Any]]]:
        """Group results according to group configuration.
        
        Args:
            results: Results to group
            group_config: GroupConfig object
            
        Returns:
            Grouped results
        """
        groups = defaultdict(list)
        
        for result in results:
            group_value = result.get(group_config.by, "Unknown")
            if group_value is None:
                group_value = "None"
            groups[str(group_value)].append(result)
        
        # Apply custom order if specified
        if group_config.order:
            ordered_groups = {}
            for key in group_config.order:
                if key in groups:
                    ordered_groups[key] = groups[key]
            # Add any remaining groups not in custom order
            for key in groups:
                if key not in ordered_groups:
                    ordered_groups[key] = groups[key]
            return ordered_groups
        
        return dict(groups)
    
    def format_view_data(self, query_result: BaseQueryResult, base_id: str, view_id: str) -> BaseViewData:
        """Format query results for specific view display.
        
        Args:
            query_result: Raw query results
            base_id: Base ID
            view_id: View ID
            
        Returns:
            Formatted view data
        """
        base = self.get_base(base_id)
        if not base:
            raise ValueError(f"Base not found: {base_id}")
        
        view = next((v for v in base.views if v.id == view_id), None)
        if not view:
            raise ValueError(f"View not found: {view_id}")
        
        if view.type == ViewType.TABLE:
            # Format as table
            columns = []
            if view.columns:
                for col in view.columns:
                    columns.append({
                        "id": col.id,
                        "header": col.header,
                        "format": col.format.value if col.format else "text",
                        "linkTo": col.linkTo
                    })
            
            rows = []
            for result in query_result.results:
                row = {}
                if view.columns:
                    for col in view.columns:
                        value = self._resolve_column_value(result, col.source)
                        row[col.id] = value
                rows.append(row)
            
            return BaseViewData(
                base_id=base_id,
                view_id=view_id,
                view_type="table",
                columns=columns,
                rows=rows,
                groups=query_result.groups,
                total_count=query_result.filtered_count,
                page_size=view.pageSize
            )
        
        elif view.type == ViewType.CARD:
            # Format as cards
            cards = []
            for result in query_result.results:
                card = {}
                if view.card:
                    if view.card.title:
                        card["title"] = self._resolve_column_value(result, view.card.title)
                    if view.card.subtitle:
                        card["subtitle"] = self._resolve_column_value(result, view.card.subtitle)
                    if view.card.image:
                        card["image"] = self._resolve_column_value(result, view.card.image)
                    if view.card.badges:
                        card["badges"] = [
                            self._resolve_column_value(result, badge) 
                            for badge in view.card.badges
                        ]
                    if view.card.footer:
                        card["footer"] = self._resolve_column_value(result, view.card.footer)
                cards.append(card)
            
            return BaseViewData(
                base_id=base_id,
                view_id=view_id,
                view_type="card",
                cards=cards,
                groups=query_result.groups,
                total_count=query_result.filtered_count,
                page_size=view.pageSize
            )
        
        else:
            raise ValueError(f"Unknown view type: {view.type}")
    
    def _resolve_column_value(self, result: Dict[str, Any], source: str) -> Any:
        """Resolve a column value from result data.
        
        Args:
            result: Result dictionary
            source: Column source specification
            
        Returns:
            Resolved value
        """
        if source.startswith('@'):
            # Computed field reference
            return result.get(source[1:])
        elif source.startswith('file.'):
            # File property
            return result.get(source)
        else:
            # Regular property
            return result.get(source)
    
    def create_base_from_graph(self, note_ids: List[str], name: str, 
                              description: Optional[str] = None) -> BaseFile:
        """Create a new base file from a selection of notes.
        
        Args:
            note_ids: List of note IDs to include
            name: Name for the new base
            description: Optional description
            
        Returns:
            Created BaseFile object
        """
        # Generate base ID from name
        base_id = name.lower().replace(' ', '-').replace('_', '-')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '-')
        
        # Create base structure
        from base_parser import BaseFile, BaseSource, BaseView, BaseColumn, ViewType
        
        base = BaseFile(
            id=base_id,
            name=name,
            version=1,
            description=description,
            source=BaseSource(
                folders=["/"],  # Root folder as we're selecting specific notes
                includeSubfolders=True,
                filters=[]  # We'll add filters to match specific notes
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
                        BaseColumn(id="modified", header="Modified", source="file.mtime", format="datetime")
                    ]
                )
            ]
        )
        
        # TODO: Add filters to match specific note_ids
        # This would require a more sophisticated filter system
        
        return base
