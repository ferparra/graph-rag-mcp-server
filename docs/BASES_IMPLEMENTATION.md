# Obsidian Bases (.base) File Support

This MCP server now provides comprehensive support for Obsidian's native `.base` file format, enabling database-style views over Markdown notes with graph-enhanced capabilities.

## Architecture

### Core Components

1. **base_parser.py** - Models and validation for .base file structure
   - Strict Pydantic models for all .base components
   - JSON/YAML parsing with validation
   - Path normalization and security checks

2. **base_manager.py** - Query execution and view management
   - Converts .base filters to ChromaDB queries
   - Executes queries against unified store
   - Formats results for table/card views
   - Manages sorting, grouping, and pagination

3. **base_expressions.py** - Expression evaluator for computed fields
   - Safe, sandboxed expression evaluation
   - Built-in functions (coalesce, clamp, daysSince, etc.)
   - Property access with dot notation
   - Circular reference detection

## MCP Tools

### Discovery & Reading
- `list_bases()` - List all .base files in vault
- `read_base(base_id)` - Get parsed .base configuration
- `validate_base(content, format)` - Validate .base syntax

### Query & View
- `execute_base_query(base_id, view_id, limit)` - Execute base query
- `get_base_view(base_id, view_id)` - Get formatted view data

### Creation & Modification
- `create_base(name, description, folders, filters)` - Create new .base file
- `update_base(base_id, updates)` - Update existing .base configuration

### Graph Integration (Unique Features)
- `base_from_graph(note_ids, name)` - Create .base from graph selection
- `enrich_base_with_graph(base_id)` - Add graph metrics as computed fields

## Graph-Enhanced Features

Our implementation extends standard .base functionality with graph capabilities:

### Graph Metadata in Queries
- Access `links_to` and `backlinks_from` in filters
- Use graph relationships in computed fields
- Sort and group by connectivity metrics

### Example Computed Fields
```json
{
  "id": "connectivity-score",
  "expr": "len(links_to) + len(backlinks_from)",
  "type": "number"
}
```

### Graph-Aware Views
Create views that leverage the knowledge graph:
- Show link counts in table columns
- Group by connectivity patterns
- Filter by graph neighborhoods

## Example .base File

See `examples/reading-list.base` for a complete example that:
- Filters books by status
- Computes reading progress
- Shows graph connectivity
- Provides both table and card views

## Integration with Unified Store

The .base implementation leverages the existing unified ChromaDB store:

1. **Vector Search** - Semantic similarity for content matching
2. **Metadata Filtering** - Efficient property-based queries
3. **Graph Relationships** - Links and tags stored as metadata
4. **Multi-hop Retrieval** - Expand results through graph traversal

## Expression Language

Supported expression features:
- **Operators**: `+ - * / % == != > >= < <= && || !`
- **Ternary**: `condition ? true_value : false_value`
- **Functions**: `coalesce`, `clamp`, `lower`, `upper`, `trim`, `concat`, `contains`, `regexMatch`, `daysSince`, `now`, `date`, `len`
- **Property Access**: `title`, `file.name`, `file.mtime`
- **Computed References**: `@field-id`

## Testing

Comprehensive test coverage in `tests/unit/`:
- `test_base_parser.py` - Model validation and parsing
- `test_base_manager.py` - Query execution and formatting
- `test_base_expressions.py` - Expression evaluation

## Usage with Claude Desktop

The MCP tools are automatically available in Claude Desktop:

```typescript
// List all bases in vault
const bases = await list_bases();

// Execute a query
const results = await execute_base_query("reading-list", "table-main");

// Create base from graph selection
const noteIds = ["Books/1984.md", "Books/Brave New World.md"];
await base_from_graph(noteIds, "Dystopian Classics");

// Enrich with graph metrics
await enrich_base_with_graph("reading-list");
```

## Benefits

1. **Native Obsidian Format** - Full compatibility with Obsidian's .base spec
2. **Graph Integration** - Unique graph-enhanced features not available elsewhere
3. **Performance** - Leverages ChromaDB's efficient indexing
4. **Real-time Updates** - File watching keeps bases synchronized
5. **MCP Protocol** - Seamless Claude Desktop integration
6. **Type Safety** - Strict validation and type checking throughout