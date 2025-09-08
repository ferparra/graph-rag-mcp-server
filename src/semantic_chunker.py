from __future__ import annotations
import re
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
# Support both package and module execution contexts
try:
    from fs_indexer import NoteDoc
except ImportError:  # When imported as part of a package
    from .fs_indexer import NoteDoc


class ChunkType(Enum):
    FRONTMATTER = "frontmatter"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    TABLE = "table"


class SemanticChunk(BaseModel):
    """A semantically meaningful chunk of text from a note."""
    id: str = Field(..., description="note_id/chunk_type_index (e.g., 'my_note.md/section_1')")
    note_id: str
    chunk_type: ChunkType
    content: str
    
    # Position and hierarchy
    position: int = Field(..., description="Order within document (0-based)")
    header_text: Optional[str] = Field(default=None)
    header_level: Optional[int] = Field(default=None, description="1-6 for H1-H6")
    parent_headers: List[str] = Field(default_factory=list, description="Full hierarchy path")
    
    # Content metadata
    start_line: int = Field(default=0)
    end_line: int = Field(default=0)
    char_start: int = Field(default=0)
    char_end: int = Field(default=0)
    
    # Semantic features
    contains_tags: List[str] = Field(default_factory=list)
    contains_links: List[str] = Field(default_factory=list)
    importance_score: float = Field(default=0.0)
    
    # Context preservation
    preceding_context: Optional[str] = Field(default=None, description="Previous header/chunk")
    following_context: Optional[str] = Field(default=None, description="Next header/chunk")


class MarkdownParser:
    """Parse markdown structure to identify semantic boundaries."""
    
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'^```[\w]*\n(.*?)^```', re.MULTILINE | re.DOTALL)
    LIST_PATTERN = re.compile(r'^(\s*)[-*+]\s+(.+)$', re.MULTILINE)
    NUMBERED_LIST_PATTERN = re.compile(r'^(\s*)\d+\.\s+(.+)$', re.MULTILINE)
    QUOTE_PATTERN = re.compile(r'^>\s+(.+)$', re.MULTILINE)
    TABLE_ROW_PATTERN = re.compile(r'^\|.+\|$', re.MULTILINE)
    WIKILINK_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')
    TAG_PATTERN = re.compile(r'(#\w[\w/-]+)')
    
    def __init__(self):
        self.lines: List[str] = []
        self.line_types: List[str] = []
    
    def parse(self, text: str) -> Dict:
        """Parse markdown text into structured components."""
        self.lines = text.split('\n')
        self.line_types = [self._classify_line(line) for line in self.lines]
        
        return {
            'headers': self._extract_headers(),
            'code_blocks': self._extract_code_blocks(text),
            'lists': self._extract_lists(),
            'quotes': self._extract_quotes(),
            'tables': self._extract_tables(),
            'paragraphs': self._extract_paragraphs()
        }
    
    def _classify_line(self, line: str) -> str:
        """Classify a single line by type."""
        line = line.strip()
        
        if not line:
            return 'empty'
        elif line.startswith('#'):
            return 'header'
        elif line.startswith('```'):
            return 'code_fence'
        elif line.startswith('>'):
            return 'quote'
        elif re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            return 'list_item'
        elif line.startswith('|') and line.endswith('|'):
            return 'table_row'
        elif set(line.strip()) <= {'-', '=', ' '}:
            return 'separator'
        else:
            return 'text'
    
    def _extract_headers(self) -> List[Dict]:
        """Extract all headers with their positions."""
        headers = []
        for i, line in enumerate(self.lines):
            match = self.HEADER_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append({
                    'level': level,
                    'text': text,
                    'line': i,
                    'full_line': line
                })
        return headers
    
    def _extract_code_blocks(self, text: str) -> List[Dict]:
        """Extract code blocks with their positions."""
        blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            start_line = text[:match.start()].count('\n')
            end_line = text[:match.end()].count('\n')
            blocks.append({
                'content': match.group(1),
                'start_line': start_line,
                'end_line': end_line,
                'full_match': match.group(0)
            })
        return blocks
    
    def _extract_lists(self) -> List[Dict[str, Any]]:
        """Extract list blocks (consecutive list items)."""
        lists: List[Dict[str, Any]] = []
        current_list: Optional[Dict[str, Any]] = None
        
        for i, line_type in enumerate(self.line_types):
            if line_type == 'list_item':
                if current_list is None:
                    current_list = {
                        'start_line': i,
                        'items': [],
                        'content': []
                    }
                current_list['items'].append(self.lines[i])
                current_list['content'].append(self.lines[i])
            else:
                if current_list is not None:
                    current_list['end_line'] = i - 1
                    current_list['text'] = '\n'.join(current_list['content'])
                    lists.append(current_list)
                    current_list = None
        
        # Handle list at end of document
        if current_list is not None:
            current_list['end_line'] = len(self.lines) - 1
            current_list['text'] = '\n'.join(current_list['content'])
            lists.append(current_list)
        
        return lists
    
    def _extract_quotes(self) -> List[Dict[str, Any]]:
        """Extract quote blocks."""
        quotes: List[Dict[str, Any]] = []
        current_quote: Optional[Dict[str, Any]] = None
        
        for i, line_type in enumerate(self.line_types):
            if line_type == 'quote':
                if current_quote is None:
                    current_quote = {
                        'start_line': i,
                        'content': []
                    }
                current_quote['content'].append(self.lines[i])
            else:
                if current_quote is not None:
                    current_quote['end_line'] = i - 1
                    current_quote['text'] = '\n'.join(current_quote['content'])
                    quotes.append(current_quote)
                    current_quote = None
        
        # Handle quote at end of document
        if current_quote is not None:
            current_quote['end_line'] = len(self.lines) - 1
            current_quote['text'] = '\n'.join(current_quote['content'])
            quotes.append(current_quote)
        
        return quotes
    
    def _extract_tables(self) -> List[Dict[str, Any]]:
        """Extract table blocks."""
        tables: List[Dict[str, Any]] = []
        current_table: Optional[Dict[str, Any]] = None
        
        for i, line_type in enumerate(self.line_types):
            if line_type == 'table_row':
                if current_table is None:
                    current_table = {
                        'start_line': i,
                        'rows': []
                    }
                current_table['rows'].append(self.lines[i])
            else:
                if current_table is not None:
                    current_table['end_line'] = i - 1
                    current_table['text'] = '\n'.join(current_table['rows'])
                    tables.append(current_table)
                    current_table = None
        
        # Handle table at end of document
        if current_table is not None:
            current_table['end_line'] = len(self.lines) - 1
            current_table['text'] = '\n'.join(current_table['rows'])
            tables.append(current_table)
        
        return tables
    
    def _extract_paragraphs(self) -> List[Dict[str, Any]]:
        """Extract paragraph blocks (consecutive text lines)."""
        paragraphs: List[Dict[str, Any]] = []
        current_para: Optional[Dict[str, Any]] = None
        
        for i, line_type in enumerate(self.line_types):
            line = self.lines[i].strip()
            
            # Skip empty lines and non-text content
            if line_type in ['empty', 'header', 'code_fence', 'list_item', 'quote', 'table_row', 'separator']:
                if current_para is not None:
                    current_para['end_line'] = i - 1
                    current_para['text'] = '\n'.join(current_para['content']).strip()
                    if current_para['text']:  # Only add non-empty paragraphs
                        paragraphs.append(current_para)
                    current_para = None
            elif line_type == 'text' and line:
                if current_para is None:
                    current_para = {
                        'start_line': i,
                        'content': []
                    }
                current_para['content'].append(line)
        
        # Handle paragraph at end of document
        if current_para is not None:
            current_para['end_line'] = len(self.lines) - 1
            current_para['text'] = '\n'.join(current_para['content']).strip()
            if current_para['text']:
                paragraphs.append(current_para)
        
        return paragraphs


class SemanticChunker:
    """Intelligent chunker that creates semantically meaningful chunks."""
    
    # Regex patterns for extracting tags and links
    TAG_PATTERN = re.compile(r'(#\w[\w/-]+)')
    WIKILINK_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')
    
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 3000,
                 merge_threshold: int = 200,
                 include_context: bool = True):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.merge_threshold = merge_threshold
        self.include_context = include_context
        self.parser = MarkdownParser()
    
    def chunk_note(self, note: NoteDoc) -> List[SemanticChunk]:
        """Create semantic chunks from a note."""
        chunks = []
        self._position_counter = 0  # Global position counter for unique IDs
        
        # Handle frontmatter as separate chunk
        if note.frontmatter:
            frontmatter_chunk = self._create_frontmatter_chunk(note)
            chunks.append(frontmatter_chunk)
        
        # Parse markdown structure
        parsed = self.parser.parse(note.text)
        
        # Create chunks based on document structure
        chunks.extend(self._create_section_chunks(note, parsed))
        
        # Merge small adjacent chunks
        chunks = self._merge_small_chunks(chunks)
        
        # Calculate importance scores
        self._calculate_importance_scores(chunks)
        
        # Add context links
        self._add_context_links(chunks)
        
        return chunks
    
    def _create_frontmatter_chunk(self, note: NoteDoc) -> SemanticChunk:
        """Create a chunk for frontmatter."""
        import yaml
        
        try:
            frontmatter_text = yaml.dump(note.frontmatter, default_flow_style=False)
        except Exception:
            frontmatter_text = str(note.frontmatter)
        
        # Ensure frontmatter_text is a string
        if not isinstance(frontmatter_text, str):
            frontmatter_text = str(frontmatter_text)
        
        chunk = SemanticChunk(
            id=f"{note.id}/chunk_{self._position_counter}",
            note_id=note.id,
            chunk_type=ChunkType.FRONTMATTER,
            content=frontmatter_text,
            position=self._position_counter,
            importance_score=0.3  # Lower importance for metadata
        )
        self._position_counter += 1
        return chunk
    
    def _create_section_chunks(self, note: NoteDoc, parsed: Dict) -> List[SemanticChunk]:
        """Create chunks based on document sections."""
        chunks = []
        headers = parsed['headers']
        
        if not headers:
            # No headers - treat as single section or split by content type
            return self._create_content_chunks(note, parsed, 0, len(self.parser.lines))
        
        # Process each section defined by headers
        for i, header in enumerate(headers):
            start_line = header['line']
            end_line = headers[i + 1]['line'] if i + 1 < len(headers) else len(self.parser.lines)
            
            section_chunks = self._create_section_with_content(
                note, parsed, header, start_line, end_line, i + 1
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _create_section_with_content(self, note: NoteDoc, parsed: Dict, header: Dict, 
                                   start_line: int, end_line: int, position: int) -> List[SemanticChunk]:
        """Create chunks for a section including its header and content."""
        chunks = []
        
        # Build header hierarchy
        parent_headers = self._build_header_hierarchy(parsed['headers'], header['level'], header['line'])
        
        # Get section content (excluding the header line)
        content_start = start_line + 1
        section_lines = self.parser.lines[start_line:end_line]
        section_text = '\n'.join(section_lines)
        
        # Extract tags and links from section
        tags = self.TAG_PATTERN.findall(section_text)
        links = self.WIKILINK_PATTERN.findall(section_text)
        
        # If section is small enough, create single chunk
        if len(section_text) <= self.max_chunk_size:
            chunk = SemanticChunk(
                id=f"{note.id}/chunk_{self._position_counter}",
                note_id=note.id,
                chunk_type=ChunkType.SECTION,
                content=section_text,
                position=self._position_counter,
                header_text=header['text'],
                header_level=header['level'],
                parent_headers=parent_headers,
                start_line=start_line,
                end_line=end_line - 1,
                contains_tags=[tag.strip('#') for tag in tags],
                contains_links=links
            )
            self._position_counter += 1
            chunks.append(chunk)
        else:
            # Large section - split by content blocks within it
            content_chunks = self._create_content_chunks(
                note, parsed, content_start, end_line, 
                header_text=header['text'], 
                header_level=header['level'],
                parent_headers=parent_headers,
                base_position=position
            )
            chunks.extend(content_chunks)
        
        return chunks
    
    def _create_content_chunks(self, note: NoteDoc, parsed: Dict, start_line: int, end_line: int,
                             header_text: Optional[str] = None, header_level: Optional[int] = None,
                             parent_headers: Optional[List[str]] = None, base_position: int = 0) -> List[SemanticChunk]:
        """Create chunks from content blocks (paragraphs, lists, code, etc.)."""
        chunks = []
        parent_headers = parent_headers or []
        
        # Find content blocks within the range
        blocks = []
        
        # Add code blocks
        for block in parsed['code_blocks']:
            if start_line <= block['start_line'] < end_line:
                blocks.append((block['start_line'], block['end_line'], 'code', block))
        
        # Add lists
        for block in parsed['lists']:
            if start_line <= block['start_line'] < end_line:
                blocks.append((block['start_line'], block['end_line'], 'list', block))
        
        # Add quotes
        for block in parsed['quotes']:
            if start_line <= block['start_line'] < end_line:
                blocks.append((block['start_line'], block['end_line'], 'quote', block))
        
        # Add tables
        for block in parsed['tables']:
            if start_line <= block['start_line'] < end_line:
                blocks.append((block['start_line'], block['end_line'], 'table', block))
        
        # Sort blocks by start line
        blocks.sort(key=lambda x: x[0])
        
        # Fill gaps with paragraphs
        current_line = start_line
        chunk_position = base_position
        
        for block_start, block_end, block_type, block_data in blocks:
            # Create paragraph chunk for content before this block
            if current_line < block_start:
                para_text = '\n'.join(self.parser.lines[current_line:block_start]).strip()
                if para_text and len(para_text) > 50:  # Skip very small paragraphs
                    para_chunk = self._create_paragraph_chunk(
                        note, para_text, chunk_position, current_line, block_start - 1,
                        header_text, header_level, parent_headers
                    )
                    chunks.append(para_chunk)
                    chunk_position += 1
            
            # Create chunk for the block
            block_chunk = self._create_block_chunk(
                note, block_type, block_data, chunk_position,
                header_text, header_level, parent_headers
            )
            chunks.append(block_chunk)
            chunk_position += 1
            current_line = block_end + 1
        
        # Handle remaining content after last block
        if current_line < end_line:
            para_text = '\n'.join(self.parser.lines[current_line:end_line]).strip()
            if para_text and len(para_text) > 50:
                para_chunk = self._create_paragraph_chunk(
                    note, para_text, chunk_position, current_line, end_line - 1,
                    header_text, header_level, parent_headers
                )
                chunks.append(para_chunk)
        
        return chunks
    
    def _create_paragraph_chunk(self, note: NoteDoc, text: str, position: int, 
                               start_line: int, end_line: int,
                               header_text: Optional[str] = None, 
                               header_level: Optional[int] = None,
                               parent_headers: Optional[List[str]] = None) -> SemanticChunk:
        """Create a paragraph chunk."""
        tags = self.TAG_PATTERN.findall(text)
        links = self.WIKILINK_PATTERN.findall(text)
        
        chunk = SemanticChunk(
            id=f"{note.id}/chunk_{self._position_counter}",
            note_id=note.id,
            chunk_type=ChunkType.PARAGRAPH,
            content=text,
            position=self._position_counter,
            header_text=header_text,
            header_level=header_level,
            parent_headers=parent_headers or [],
            start_line=start_line,
            end_line=end_line,
            contains_tags=[tag.strip('#') for tag in tags],
            contains_links=links
        )
        self._position_counter += 1
        return chunk
    
    def _create_block_chunk(self, note: NoteDoc, block_type: str, block_data: Dict[str, Any], position: int,
                           header_text: Optional[str] = None, 
                           header_level: Optional[int] = None,
                           parent_headers: Optional[List[str]] = None) -> SemanticChunk:
        """Create a chunk for a specific block type."""
        chunk_type_map = {
            'code': ChunkType.CODE_BLOCK,
            'list': ChunkType.LIST,
            'quote': ChunkType.QUOTE,
            'table': ChunkType.TABLE
        }
        
        text = block_data.get('text', block_data.get('content', ''))
        if not isinstance(text, str):
            text = str(text)
            
        tags = self.TAG_PATTERN.findall(text)
        links = self.WIKILINK_PATTERN.findall(text)
        
        chunk = SemanticChunk(
            id=f"{note.id}/chunk_{self._position_counter}",
            note_id=note.id,
            chunk_type=chunk_type_map[block_type],
            content=text,
            position=self._position_counter,
            header_text=header_text,
            header_level=header_level,
            parent_headers=parent_headers or [],
            start_line=block_data['start_line'],
            end_line=block_data['end_line'],
            contains_tags=[tag.strip('#') for tag in tags],
            contains_links=links
        )
        self._position_counter += 1
        return chunk
    
    def _build_header_hierarchy(self, headers: List[Dict], current_level: int, current_line: int) -> List[str]:
        """Build the hierarchy of parent headers for a given header."""
        hierarchy = []
        
        for header in headers:
            if header['line'] >= current_line:
                break
            if header['level'] < current_level:
                # Remove headers at same or deeper level
                while hierarchy and hierarchy[-1]['level'] >= header['level']:
                    hierarchy.pop()
                hierarchy.append(header)
        
        return [h['text'] for h in hierarchy]
    
    def _merge_small_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Merge small adjacent chunks of similar type."""
        if not chunks:
            return chunks
        
        merged = [chunks[0]]
        
        for chunk in chunks[1:]:
            last_chunk = merged[-1]
            
            # Check if chunks should be merged
            should_merge = (
                len(chunk.content) < self.merge_threshold and
                len(last_chunk.content) < self.merge_threshold and
                chunk.chunk_type == last_chunk.chunk_type and
                chunk.header_level == last_chunk.header_level and
                chunk.header_text == last_chunk.header_text
            )
            
            if should_merge:
                # Merge the chunks
                merged_content = last_chunk.content + '\n\n' + chunk.content
                merged_chunk = SemanticChunk(
                    id=last_chunk.id,
                    note_id=last_chunk.note_id,
                    chunk_type=last_chunk.chunk_type,
                    content=merged_content,
                    position=last_chunk.position,
                    header_text=last_chunk.header_text,
                    header_level=last_chunk.header_level,
                    parent_headers=last_chunk.parent_headers,
                    start_line=last_chunk.start_line,
                    end_line=chunk.end_line,
                    contains_tags=list(set(last_chunk.contains_tags + chunk.contains_tags)),
                    contains_links=list(set(last_chunk.contains_links + chunk.contains_links))
                )
                merged[-1] = merged_chunk
            else:
                merged.append(chunk)
        
        return merged
    
    def _calculate_importance_scores(self, chunks: List[SemanticChunk]):
        """Calculate importance scores for chunks."""
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            score = 0.5  # Base score
            
            # Header level importance (higher level = more important)
            if chunk.header_level:
                score += (7 - chunk.header_level) * 0.1  # H1=0.6, H2=0.5, etc.
            
            # Position importance (intro and conclusion weighted higher)
            position_ratio = i / max(total_chunks - 1, 1)
            if position_ratio < 0.2 or position_ratio > 0.8:  # First 20% or last 20%
                score += 0.1
            
            # Content type importance
            type_scores = {
                ChunkType.SECTION: 0.1,
                ChunkType.FRONTMATTER: -0.2,
                ChunkType.CODE_BLOCK: 0.05,
                ChunkType.LIST: 0.0,
                ChunkType.QUOTE: 0.0,
                ChunkType.TABLE: 0.05,
                ChunkType.PARAGRAPH: 0.0
            }
            score += type_scores.get(chunk.chunk_type, 0)
            
            # Link density (more links = more important)
            if chunk.contains_links:
                score += min(len(chunk.contains_links) * 0.05, 0.2)
            
            # Tag presence
            if chunk.contains_tags:
                score += min(len(chunk.contains_tags) * 0.03, 0.15)
            
            # Content length (moderate length preferred)
            content_len = len(chunk.content)
            if 200 <= content_len <= 1000:
                score += 0.05
            elif content_len < 100:
                score -= 0.1
            
            chunk.importance_score = max(0.0, min(1.0, score))
    
    def _add_context_links(self, chunks: List[SemanticChunk]):
        """Add context links between adjacent chunks."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                chunk.preceding_context = f"{prev_chunk.chunk_type.value}: {prev_chunk.content[:100]}..."
            
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                chunk.following_context = f"{next_chunk.chunk_type.value}: {next_chunk.content[:100]}..."
