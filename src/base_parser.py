"""Parser and models for Obsidian .base files."""

from __future__ import annotations
import json
import yaml
from pathlib import Path
from typing import List, Optional, Any, Literal, cast, Annotated
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)

# Enums for strict validation
class FilterOperator(str, Enum):
    """Supported filter operators."""
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NIN = "nin"
    CONTAINS = "contains"
    NCONTAINS = "ncontains"
    EXISTS = "exists"
    NEXISTS = "nexists"
    REGEX = "regex"

class ViewType(str, Enum):
    """Supported view types."""
    TABLE = "table"
    CARD = "card"

class ColumnFormat(str, Enum):
    """Column display formats."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    DATETIME = "datetime"
    PROGRESS = "progress"
    BADGE = "badge"
    URL = "url"

class SortDirection(str, Enum):
    """Sort directions."""
    ASC = "asc"
    DESC = "desc"

class NullPosition(str, Enum):
    """Null value positions in sorting."""
    FIRST = "first"
    LAST = "last"

class ComputedType(str, Enum):
    """Computed field data types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"


# Models for .base file structure
class BaseFilter(BaseModel):
    """Filter condition for selecting notes."""
    property: str = Field(..., min_length=1, description="Property to filter on")
    op: FilterOperator = Field(..., description="Filter operator")
    value: Optional[Any] = Field(None, description="Value to compare against")
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v, info):
        """Validate value based on operator."""
        op = info.data.get('op')
        if op in [FilterOperator.EXISTS, FilterOperator.NEXISTS]:
            # These operators don't need a value
            return None
        elif op in [FilterOperator.IN, FilterOperator.NIN]:
            # These need list values
            if not isinstance(v, list):
                return [v] if v is not None else []
        return v
    
    @field_validator('property')
    @classmethod
    def validate_property(cls, v):
        """Validate property name format."""
        # Allow file.* properties and regular properties
        if v.startswith('file.'):
            allowed_file_props = ['name', 'basename', 'path', 'ext', 'size', 'ctime', 'mtime']
            prop = v.replace('file.', '')
            if prop not in allowed_file_props:
                logger.warning(f"Unknown file property: {v}")
        return v


class BaseSource(BaseModel):
    """Source configuration for selecting notes."""
    folders: List[str] = Field(..., min_length=1, description="Folders to search")
    includeSubfolders: bool = Field(True, description="Include subfolders")
    filters: List[BaseFilter] = Field(default_factory=list, description="Filter conditions")
    
    @field_validator('folders')
    @classmethod
    def normalize_paths(cls, v):
        """Normalize folder paths to use forward slashes."""
        return [path.replace('\\', '/') for path in v]
    
    @field_validator('folders')
    @classmethod
    def validate_paths(cls, v):
        """Ensure no path traversal attempts."""
        for path in v:
            if '..' in path:
                raise ValueError(f"Path traversal not allowed: {path}")
        return v


class BaseColumn(BaseModel):
    """Column configuration for table view."""
    id: str = Field(..., pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", description="Column ID")
    header: str = Field(..., min_length=1, description="Column header text")
    source: str = Field(..., min_length=1, description="Data source (property or computed)")
    format: Annotated[Optional[ColumnFormat], Field(description="Display format")] = None
    min: Annotated[Optional[float], Field(description="Minimum value for progress")] = None
    max: Annotated[Optional[float], Field(description="Maximum value for progress")] = None
    linkTo: Annotated[Optional[Literal["file", "none"]], Field(description="Link behavior")] = None
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        """Validate source reference."""
        # Sources can be:
        # - Regular property: "title", "author"
        # - File property: "file.name", "file.mtime"
        # - Computed reference: "@healthScore"
        if v.startswith('@') and len(v) < 2:
            raise ValueError(f"Invalid computed reference: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_progress_bounds(self):
        """Validate min/max for progress format."""
        if self.format == ColumnFormat.PROGRESS:
            if self.min is None or self.max is None:
                raise ValueError("Progress format requires min and max values")
            if self.min >= self.max:
                raise ValueError("Progress min must be less than max")
        return self


class CardConfig(BaseModel):
    """Configuration for card view display."""
    title: Optional[str] = Field(None, description="Property for card title")
    subtitle: Optional[str] = Field(None, description="Property for card subtitle")
    image: Optional[str] = Field(None, description="Property for card image")
    badges: List[str] = Field(default_factory=list, description="Properties to show as badges")
    footer: Optional[str] = Field(None, description="Property for card footer")


class SortConfig(BaseModel):
    """Sort configuration."""
    by: str = Field(..., min_length=1, description="Column ID or property to sort by")
    dir: SortDirection = Field(SortDirection.ASC, description="Sort direction")
    nulls: NullPosition = Field(NullPosition.LAST, description="Null position")


class GroupConfig(BaseModel):
    """Group configuration."""
    by: str = Field(..., min_length=1, description="Property to group by")
    order: Optional[List[str]] = Field(None, description="Custom group order")


class BaseView(BaseModel):
    """View configuration."""
    id: str = Field(..., pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", description="View ID")
    name: str = Field(..., min_length=1, description="View name")
    type: ViewType = Field(..., description="View type")
    columns: Annotated[Optional[List[BaseColumn]], Field(description="Table columns")] = None
    card: Annotated[Optional[CardConfig], Field(description="Card configuration")] = None
    group: Annotated[Optional[GroupConfig], Field(description="Grouping configuration")] = None
    sort: List[SortConfig] = Field(default_factory=list, description="Sort configuration")
    pageSize: Annotated[Optional[int], Field(ge=1, le=1000, description="Page size")] = None
    
    @model_validator(mode='after')
    def validate_view_config(self):
        """Validate view-specific configuration."""
        if self.type == ViewType.TABLE:
            if not self.columns:
                raise ValueError("Table view requires columns")
        elif self.type == ViewType.CARD:
            if self.columns:
                logger.warning("Card view doesn't use columns configuration")
        return self
    
    @model_validator(mode='after')
    def validate_sort_references(self):
        """Validate that sort references exist."""
        if self.type == ViewType.TABLE and self.columns:
            column_ids = {col.id for col in self.columns}
            column_sources = {col.source for col in self.columns}
            
            for sort_cfg in self.sort:
                # Sort can reference column ID, source, or computed field
                if not (sort_cfg.by in column_ids or 
                        sort_cfg.by in column_sources or
                        sort_cfg.by.startswith('@') or
                        sort_cfg.by.startswith('file.')):
                    logger.warning(f"Sort references unknown field: {sort_cfg.by}")
        return self


class ComputedField(BaseModel):
    """Computed field definition."""
    id: str = Field(..., pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", description="Computed field ID")
    expr: str = Field(..., min_length=1, description="Expression to evaluate")
    type: ComputedType = Field(..., description="Result data type")
    
    @field_validator('expr')
    @classmethod
    def validate_expression(cls, v):
        """Basic expression validation."""
        # Check for obvious syntax errors
        if v.count('(') != v.count(')'):
            raise ValueError("Unbalanced parentheses in expression")
        if v.count('[') != v.count(']'):
            raise ValueError("Unbalanced brackets in expression")
        # More validation would be done by the expression evaluator
        return v


class BaseFile(BaseModel):
    """Complete .base file structure."""
    schema_: Optional[str] = None
    id: str = Field(..., pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", description="Base ID")
    name: str = Field(..., min_length=1, description="Base name")
    version: Annotated[int, Field(ge=1, description="Version number")]
    description: Optional[str] = Field(None, description="Base description")
    source: BaseSource = Field(..., description="Source configuration")
    computed: List[ComputedField] = Field(default_factory=list, description="Computed fields")
    views: List[BaseView] = Field(..., min_length=1, description="View configurations")
    
    @model_validator(mode='after')
    def validate_unique_ids(self):
        """Ensure all IDs are unique within the file."""
        all_ids = [self.id]
        all_ids.extend(cf.id for cf in self.computed)
        all_ids.extend(v.id for v in self.views)
        
        for view in self.views:
            if view.columns:
                all_ids.extend(col.id for col in view.columns)
        
        if len(all_ids) != len(set(all_ids)):
            # Find duplicates
            seen = set()
            dupes = set()
            for id_ in all_ids:
                if id_ in seen:
                    dupes.add(id_)
                seen.add(id_)
            raise ValueError(f"Duplicate IDs found: {dupes}")
        
        return self
    
    @model_validator(mode='after')
    def validate_computed_references(self):
        """Validate computed field references."""
        computed_ids = {cf.id for cf in self.computed}
        
        # Check that computed references in columns are valid
        for view in self.views:
            if view.columns:
                for col in view.columns:
                    if col.source.startswith('@'):
                        ref_id = col.source[1:]
                        if ref_id not in computed_ids:
                            raise ValueError(f"Column references unknown computed field: {col.source}")
            
            # Check card config references
            if view.card:
                for field in [view.card.title, view.card.subtitle, 
                             view.card.image, view.card.footer]:
                    if field and field.startswith('@'):
                        ref_id = field[1:]
                        if ref_id not in computed_ids:
                            raise ValueError(f"Card references unknown computed field: {field}")
        
        return self


class BaseParser:
    """Parser for .base files."""
    
    @staticmethod
    def parse_file(path: Path) -> BaseFile:
        """Parse a .base file from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Base file not found: {path}")
        
        content = path.read_text(encoding='utf-8')
        
        # Determine format by extension or content
        if path.suffix.lower() in ['.yaml', '.yml']:
            return BaseParser.parse_yaml(content)
        else:
            # Try JSON first, fall back to YAML
            try:
                return BaseParser.parse_json(content)
            except json.JSONDecodeError:
                return BaseParser.parse_yaml(content)
    
    @staticmethod
    def parse_json(content: str) -> BaseFile:
        """Parse JSON content into BaseFile model."""
        try:
            data = json.loads(content)
            # Map $schema -> schema_ for validation
            if isinstance(data, dict) and "$schema" in data:
                data["schema_"] = data.pop("$schema")
            return BaseFile.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Invalid base file structure: {e}")
    
    @staticmethod
    def parse_yaml(content: str) -> BaseFile:
        """Parse YAML content into BaseFile model."""
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                raise ValueError("YAML must contain a dictionary")
            # Map $schema -> schema_ for validation
            if "$schema" in data:
                data["schema_"] = data.pop("$schema")
            return BaseFile.model_validate(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
        except Exception as e:
            raise ValueError(f"Invalid base file structure: {e}")
    
    @staticmethod
    def to_json(base: BaseFile, indent: int = 2) -> str:
        """Convert BaseFile to JSON string."""
        data = base.model_dump(exclude_none=True)
        # Map schema_ -> $schema for serialization
        if "schema_" in data:
            data["$schema"] = data.pop("schema_")
        return json.dumps(data, indent=indent)
    
    @staticmethod
    def to_yaml(base: BaseFile) -> str:
        """Convert BaseFile to YAML string."""
        data = base.model_dump(exclude_none=True)
        # Map schema_ -> $schema for serialization
        if "schema_" in data:
            data["$schema"] = data.pop("schema_")
        dumped = yaml.dump(data, default_flow_style=False, sort_keys=False)
        return cast(str, dumped)
    
    @staticmethod
    def validate(content: str, format: Optional[Literal["json", "yaml"]] = None) -> tuple[bool, Optional[str]]:
        """Validate base file content without raising exceptions.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if format == "yaml":
                BaseParser.parse_yaml(content)
            elif format == "json":
                BaseParser.parse_json(content)
            else:
                # Auto-detect
                try:
                    BaseParser.parse_json(content)
                except Exception:
                    BaseParser.parse_yaml(content)
            return True, None
        except Exception as e:
            return False, str(e)