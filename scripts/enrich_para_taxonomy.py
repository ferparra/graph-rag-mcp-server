#!/usr/bin/env python3
"""
PARA Taxonomy Enrichment Script

This script analyzes Obsidian notes and enriches them with PARA taxonomy
metadata using DSPy and the MCP server tools. It identifies the taxonomy
of tags and applies the PARA system (Projects, Areas, Resources, Archive)
to create more meaningful graph relationships.

The script:
1. Uses DSPy with Gemini for intelligent classification
2. Leverages MCP server tools for vault interaction  
3. Implements safety checks to preserve existing properties
4. Only improves frontmatter without touching content
"""

import sys
import typer
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
import dspy
from src.config import settings
from src.dspy_rag import GeminiLM
from src.mcp_server import app_state

# Add parent directory to path for imports
project_root: Path = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


app = typer.Typer(help="PARA Taxonomy Enrichment for Obsidian Vault")

# DSPy Signatures for Classification
class PARAClassifier(dspy.Signature):
    """Classify note content and location into PARA system categories."""
    content = dspy.InputField(desc="Note content (first 1000 chars)")
    file_path = dspy.InputField(desc="File path and folder structure")
    existing_tags = dspy.InputField(desc="Current tags in the note")
    para_type = dspy.OutputField(desc="Primary PARA type: project|area|resource|archive")
    para_category = dspy.OutputField(desc="Specific category within PARA type (e.g., health/fitness)")
    confidence = dspy.OutputField(desc="Confidence score 0.0-1.0 for classification")
    reasoning = dspy.OutputField(desc="Brief explanation for classification")

class ConceptExtractor(dspy.Signature):
    """Extract key concepts and suggest hierarchical tags."""
    content = dspy.InputField(desc="Note content")
    existing_vault_tags = dspy.InputField(desc="Sample of existing tags in vault")
    existing_note_tags = dspy.InputField(desc="Current tags in this note")
    key_concepts = dspy.OutputField(desc="List of 3-5 key concepts from content")
    suggested_tags = dspy.OutputField(desc="Hierarchical tags to add (e.g., #area/health/nutrition)")
    related_topics = dspy.OutputField(desc="Topics that could link to other notes")

class RelationshipFinder(dspy.Signature):
    """Find potential relationships between notes."""
    note_content = dspy.InputField(desc="Current note content")
    note_concepts = dspy.InputField(desc="Key concepts from current note")
    similar_notes = dspy.InputField(desc="Similar notes from vector search")
    potential_links = dspy.OutputField(desc="Note titles that should be linked")
    relationship_types = dspy.OutputField(desc="Types of relationships (supports, contradicts, expands)")
    connection_strength = dspy.OutputField(desc="Strength of connections 0.0-1.0")

class PARAEnricher:
    """PARA taxonomy enrichment engine using DSPy."""
    
    def __init__(self) -> None:
        """Initialize the enricher with DSPy modules."""
        try:
            # Configure DSPy with Gemini
            self.lm = GeminiLM(model=settings.gemini_model)
            dspy.configure(lm=self.lm)
            
            # Initialize DSPy modules
            self.para_classifier = dspy.Predict(PARAClassifier)
            self.concept_extractor = dspy.Predict(ConceptExtractor)
            self.relationship_finder = dspy.Predict(RelationshipFinder)
            
            print("✅ DSPy modules initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize DSPy: {e}")
            raise
    
    def _safe_text(self, value: Any, default: str = "") -> str:
        try:
            if value is None:
                return default
            return str(value)
        except Exception:
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value)
            s_clean = s.strip()
            if s_clean.replace('.', '', 1).isdigit():
                return float(s_clean)
            return default
        except Exception:
            return default

    def _resolve_prediction(self, maybe_prediction: Any) -> Any:
        """Resolve a possibly-async DSPy prediction synchronously when needed."""
        try:
            if hasattr(maybe_prediction, "__await__"):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(maybe_prediction)
        except Exception:
            pass
        return maybe_prediction
    
    def get_vault_tag_sample(self, limit: int = 50) -> List[str]:
        """Get a sample of existing tags from the vault for context."""
        try:
            # Get tags from RDF graph
            tags = app_state.graph_store.get_all_tags(limit=limit)
            return [tag.get('tag', '') for tag in tags if tag.get('tag')]
        except Exception:
            # Fallback: extract from a few notes
            try:
                notes = app_state.chroma_store.get_all_notes(limit=20)
                tags = set()
                for note in notes:
                    note_tags = note.get('meta', {}).get('tags', [])
                    if isinstance(note_tags, str):
                        note_tags = note_tags.split(',') if note_tags else []
                    tags.update(note_tags)
                return list(tags)[:limit]
            except Exception:
                return []
    
    def classify_note(self, note_path: str, content: str, existing_tags: List[str]) -> Dict[str, Any]:
        """Classify a note using PARA taxonomy."""
        try:
            # Truncate content for efficiency
            content_preview = content[:1000] if content else ""
            
            # Get classification
            result = self._resolve_prediction(self.para_classifier(
                content=content_preview,
                file_path=note_path,
                existing_tags=", ".join(existing_tags)
            ))
            
            # Clean markdown from reasoning
            import re
            reasoning = self._safe_text(getattr(result, 'reasoning', ''))
            reasoning = re.sub(r'\*{1,2}|_{1,2}|~{1,2}|`', '', reasoning)
            reasoning = re.sub(r'\s+', ' ', reasoning).strip()
            
            return {
                "para_type": self._safe_text(getattr(result, 'para_type', 'resource')).lower().strip(),
                "para_category": self._safe_text(getattr(result, 'para_category', 'uncategorized')).strip(),
                "confidence": self._safe_float(getattr(result, 'confidence', 0.5), 0.5),
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"Classification error for {note_path}: {e}")
            return {
                "para_type": "resource",  # Default classification
                "para_category": "uncategorized",
                "confidence": 0.1,
                "reasoning": f"Auto-classified due to error: {e}"
            }
    
    def extract_concepts(self, content: str, vault_tags: List[str], note_tags: List[str]) -> Dict[str, Any]:
        """Extract key concepts and suggest tags."""
        try:
            result = self._resolve_prediction(self.concept_extractor(
                content=content[:1500],
                existing_vault_tags=", ".join(vault_tags[:30]),
                existing_note_tags=", ".join(note_tags)
            ))
            
            # Parse the outputs - handle various separators intelligently
            import re
            
            # First, replace newlines that are within sentences with spaces
            raw_text = self._safe_text(getattr(result, 'key_concepts', ''))
            # Replace newlines that aren't preceded by sentence endings with space
            raw_text = re.sub(r'(?<![.!?])\n', ' ', raw_text)
            
            # Now split on actual delimiters (numbered items or bullet points)
            # Look for patterns like "1.", "2.", "•", "-" at the start of lines
            concepts_raw = re.split(r'\n(?=\d+\.|\n[-•])', raw_text)
            
            # If no numbered/bulleted list found, try comma separation
            if len(concepts_raw) == 1:
                concepts_raw = raw_text.split(',')
            
            concepts = []
            for c in concepts_raw:
                c = c.strip()
                # Remove numbering, bullets, and dashes at the start
                c = re.sub(r'^[\d]+\.\s*|^[-•]\s*', '', c)
                # Remove markdown formatting characters
                c = re.sub(r'\*{1,2}|_{1,2}|~{1,2}|`', '', c)
                # Clean up any resulting double spaces
                c = re.sub(r'\s+', ' ', c).strip()
                # Remove trailing punctuation that might be left over
                c = c.rstrip('.,;')
                if c and len(c) > 5:  # Filter out very short fragments
                    concepts.append(c)
            concepts = concepts[:5]
            
            # Parse suggested tags - handle both comma and newline separators
            # Also remove # prefix if present
            raw_tags = self._safe_text(getattr(result, 'suggested_tags', '')).replace('\n', ',').split(',')
            suggested_tags = []
            for tag in raw_tags:
                tag = tag.strip()
                if tag:
                    # Remove leading # if present
                    if tag.startswith('#'):
                        tag = tag[1:]
                    # Only add valid tags
                    if tag and not tag.isspace():
                        suggested_tags.append(tag)
            suggested_tags = suggested_tags[:10]
            
            # Parse related topics - handle various separators intelligently
            raw_text = self._safe_text(getattr(result, 'related_topics', ''))
            # Replace newlines that aren't preceded by sentence endings with space
            raw_text = re.sub(r'(?<![.!?])\n', ' ', raw_text)
            
            # Split on actual delimiters
            topics_raw = re.split(r'\n(?=\d+\.|\n[-•])', raw_text)
            
            # If no numbered/bulleted list found, try comma separation
            if len(topics_raw) == 1:
                topics_raw = raw_text.split(',')
            
            topics = []
            for t in topics_raw:
                t = t.strip()
                # Remove numbering, bullets, and dashes at the start
                t = re.sub(r'^[\d]+\.\s*|^[-•]\s*', '', t)
                # Remove markdown formatting characters
                t = re.sub(r'\*{1,2}|_{1,2}|~{1,2}|`', '', t)
                # Clean up any resulting double spaces
                t = re.sub(r'\s+', ' ', t).strip()
                # Remove trailing punctuation
                t = t.rstrip('.,;')
                if t and len(t) > 5:  # Filter out very short fragments
                    topics.append(t)
            topics = topics[:5]
            
            return {
                "key_concepts": concepts,
                "suggested_tags": suggested_tags,
                "related_topics": topics
            }
            
        except Exception as e:
            print(f"Concept extraction error: {e}")
            return {
                "key_concepts": [],
                "suggested_tags": [],
                "related_topics": []
            }
    
    def validate_note_exists(self, note_title: str) -> bool:
        """Check if a note with this title exists in the vault."""
        try:
            # Use ChromaDB to quickly check if a note with this title exists
            # This is much faster than file system traversal
            results = app_state.chroma_store._collection().query(
                query_texts=[""],  # Empty query
                n_results=1,
                where={"title": note_title}  # Filter by exact title
            )
            
            # If we found any results with this title, the note exists
            if results and results.get('ids') and len(results['ids'][0]) > 0:
                return True
            
            # Also check for case-insensitive match using search
            search_results = app_state.searcher.search(note_title, k=1)
            for result in search_results:
                if result.get('meta', {}).get('title', '').lower() == note_title.lower():
                    return True
            
            return False
        except Exception:
            return False
    
    def find_relationships(self, content: str, concepts: List[str], current_note_title: Optional[str] = None) -> Dict[str, Any]:
        """Find potential relationships with other notes."""
        try:
            # Search for similar notes
            similar_notes = app_state.searcher.search(" ".join(concepts[:3]), k=5)
            
            # Format similar notes for context and collect valid titles
            similar_context = []
            valid_note_titles = []
            for note in similar_notes:
                note_meta = note.get('meta', {})
                title = note_meta.get('title', 'Unknown')
                similar_context.append(f"'{title}': {note.get('text', '')[:100]}")
                if title and title != 'Unknown':
                    valid_note_titles.append(title)
            
            if not similar_context:
                return {"potential_links": [], "relationship_types": [], "connection_strength": 0.0}
            
            result = self._resolve_prediction(self.relationship_finder(
                note_content=content[:1000],
                note_concepts=", ".join(concepts),
                similar_notes="; ".join(similar_context[:3])
            ))
            
            # Parse and validate links
            raw_links_raw = self._safe_text(getattr(result, 'potential_links', ''))
            raw_links = [link.strip() for link in raw_links_raw.split(',') if link.strip()]
            
            # Format as wikilinks and validate existence
            validated_links = []
            for link in raw_links:
                # Minimal cleanup - just strip whitespace
                link = link.strip()
                
                # If already formatted as wikilink, extract the title
                if link.startswith('[[') and link.endswith(']]'):
                    link = link[2:-2]
                
                # Skip empty or generic responses
                if not link or link.lower() in ['unknown', 'none', 'n/a']:
                    continue
                
                # Skip self-references
                if current_note_title and link.lower() == current_note_title.lower():
                    continue
                
                # Check if this note actually exists in similar notes or validate it
                if link in valid_note_titles or self.validate_note_exists(link):
                    # Format as Obsidian wikilink - preserve exact note name
                    validated_links.append(f"[[{link}]]")
            
            rel_raw = self._safe_text(getattr(result, 'relationship_types', ''))
            relationships = [r.strip() for r in rel_raw.split(',') if r.strip()]
            strength = self._safe_float(getattr(result, 'connection_strength', 0.0), 0.0)
            
            return {
                "potential_links": validated_links[:5],
                "relationship_types": relationships[:5], 
                "connection_strength": strength
            }
            
        except Exception as e:
            print(f"Relationship finding error: {e}")
            return {"potential_links": [], "relationship_types": [], "connection_strength": 0.0}
    
    def enrich_note_properties(self, note_path: str, dry_run: bool = True) -> Optional[Dict[str, Any]]:
        """Enrich a single note with PARA taxonomy and relationships."""
        try:
            # Check if file exists and is readable
            path: Path = Path(note_path)
            # Protect the deterministic test corpus from any mutation
            try:
                from src.fs_indexer import is_protected_test_content
                if is_protected_test_content(path):
                    print(f"⏭️  Skipping protected test content: {note_path}")
                    return None
            except Exception:
                pass
            if not path.exists():
                print(f"❌ File not found: {note_path}")
                return None
            
            if path.stat().st_size == 0:
                print(f"⚠️  Skipping empty file: {note_path}")
                return None
            
            # Read note using helper function (since MCP tools aren't directly callable)
            from src.mcp_server import _load_note
            note_info = _load_note(note_path)
            
            print(f"📝 Processing: {note_info.title}")
            
            # Skip if no content to analyze
            if not note_info.content or not note_info.content.strip():
                print(f"⚠️  Skipping note with no content: {note_path}")
                return None
            
            # Get existing properties
            existing_props = note_info.frontmatter or {}
            
            # Extract existing tags
            existing_tags = []
            if 'tags' in existing_props:
                tags = existing_props['tags']
                if isinstance(tags, list):
                    # Handle list of tags (could be multi-line strings)
                    for tag_item in tags:
                        if isinstance(tag_item, str):
                            # Split on newlines and commas
                            for t in tag_item.replace('\n', ',').split(','):
                                t = t.strip()
                                if t and t not in ["'", '"', '-']:
                                    # Remove quotes and # prefix
                                    t = t.strip("'\"").lstrip('#')
                                    if t:
                                        existing_tags.append(t)
                elif isinstance(tags, str):
                    # Handle string tags
                    for t in tags.replace('\n', ',').split(','):
                        t = t.strip()
                        if t and t not in ["'", '"', '-']:
                            t = t.strip("'\"").lstrip('#')
                            if t:
                                existing_tags.append(t)
            
            # Also extract inline tags from content
            import re
            inline_tags = re.findall(r'#(\w+(?:/\w+)*)', note_info.content)
            all_tags: list[str] = list(set(existing_tags + inline_tags))
            
            # Get vault context
            vault_tags: list[str] = self.get_vault_tag_sample()
            
            # Perform analysis
            classification = self.classify_note(note_path, note_info.content, all_tags)
            concepts = self.extract_concepts(note_info.content, vault_tags, all_tags)
            relationships = self.find_relationships(note_info.content, concepts['key_concepts'], note_info.title)
            
            # Build enriched properties
            enriched_props = existing_props.copy()
            
            # Add PARA classification
            enriched_props.update({
                'para_type': classification['para_type'],
                'para_category': classification['para_category'],
                'para_confidence': classification['confidence'],
                'para_reasoning': classification['reasoning']
            })
            
            # Add concept analysis
            if concepts['key_concepts']:
                enriched_props['key_concepts'] = concepts['key_concepts']
            if concepts['related_topics']:
                enriched_props['related_topics'] = concepts['related_topics']
            
            # Build comprehensive tag list including PARA taxonomy
            # Clean existing tags (remove # if present)
            clean_existing = []
            for tag in existing_tags:
                tag = tag.strip()
                if tag.startswith('#'):
                    tag = tag[1:]
                if tag:
                    clean_existing.append(tag)
            
            current_tags = set(clean_existing)
            
            # Add PARA tags
            para_tags = set()
            if classification['para_type']:
                para_tags.add(f"para/{classification['para_type']}")
            if classification['para_category']:
                # Clean the category path
                category = classification['para_category'].replace(' ', '-').lower()
                para_tags.add(f"para/{classification['para_type']}/{category}")
            
            # Add suggested tags
            new_tags = set(concepts['suggested_tags']) if concepts['suggested_tags'] else set()
            
            # Combine all tags
            all_tags_combined = sorted(list(current_tags | new_tags | para_tags))
            enriched_props['tags'] = all_tags_combined
            
            # Add relationship info
            if relationships['potential_links']:
                enriched_props['potential_links'] = relationships['potential_links']
                enriched_props['connection_strength'] = relationships['connection_strength']
            
            # Add enrichment metadata
            enriched_props.update({
                'enrichment_version': '1.0',
                'last_enriched': datetime.now().isoformat(),
                'enrichment_model': settings.gemini_model
            })
            
            # Show the changes
            print(f"  🎯 PARA: {classification['para_type']} -> {classification['para_category']}")
            print(f"  💡 Concepts: {', '.join(concepts['key_concepts'][:3])}")
            print(f"  🏷️  New tags: {len(concepts['suggested_tags'])} suggested")
            print(f"  🔗 Links: {len(relationships['potential_links'])} potential")
            if relationships['potential_links']:
                print(f"     Links found: {relationships['potential_links']}")
            
            if not dry_run:
                # Apply changes - simplified approach
                import frontmatter
                
                try:
                    # Load note with frontmatter
                    with open(note_path, 'r', encoding='utf-8') as f:
                        post = frontmatter.load(f)
                    
                    # Clean enriched properties to ensure all values are YAML-safe
                    clean_props = {}
                    for key, value in enriched_props.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            clean_props[key] = value
                        elif isinstance(value, (list, tuple)):
                            # Ensure all list items are YAML-safe strings
                            clean_props[key] = [str(item) for item in value]
                        else:
                            clean_props[key] = str(value)
                    
                    # Update metadata (merge preserves existing)
                    post.metadata.update(clean_props)
                    
                    # Write back to file using string output
                    output = frontmatter.dumps(post)
                    with open(note_path, 'w', encoding='utf-8') as f:
                        f.write(output)
                    
                    # Update indices - convert NoteInfo to NoteDoc for upsert
                    from src.fs_indexer import parse_note
                    updated_note_doc = parse_note(Path(note_path))
                    app_state.chroma_store.upsert_note(updated_note_doc)
                    app_state.graph_store.upsert_note(updated_note_doc)
                    
                    print("  ✅ Properties updated successfully")
                    
                except Exception as e:
                    print(f"  ❌ Failed to update properties: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
                
                print("  ✅ Properties updated")
            else:
                print("  🧪 Dry run - no changes applied")
            
            return {
                'note_path': note_path,
                'title': note_info.title,
                'classification': classification,
                'concepts': concepts,
                'relationships': relationships,
                'enriched_properties': enriched_props,
                'original_properties': existing_props
            }
            
        except Exception as e:
            print(f"❌ Error processing {note_path}: {e}")
            return None

@app.command()
def enrich(
    note_paths: List[str] = typer.Argument(None, help="Specific note paths to enrich"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview changes without applying them"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of notes to process"),
    filter_folder: str = typer.Option(None, "--folder", help="Only process notes in this folder"),
    filter_para: str = typer.Option(None, "--para-type", help="Only process notes of this PARA type")
):
    """Enrich notes with PARA taxonomy and enhanced metadata."""
    
    print("🚀 Starting PARA Taxonomy Enrichment")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY CHANGES'}")
    
    try:
        # Initialize enricher
        enricher = PARAEnricher()
        
        # Get notes to process
        if note_paths:
            # Process specific paths
            notes_to_process = note_paths
        else:
            # Get all notes from ChromaDB and filter for non-empty files
            print("📚 Discovering notes in vault...")
            all_notes = app_state.chroma_store.get_all_notes(limit=limit * 10)  # Get more to filter from
            
            notes_to_process = []
            for note in all_notes:
                note_meta = note.get('meta', {})
                note_path = note_meta.get('path', '')
                
                if not note_path:
                    continue
                
                # Apply filters
                if filter_folder and filter_folder not in note_path:
                    continue
                
                # Check if file exists and has content
                try:
                    path = Path(note_path)
                    if not path.exists():
                        continue
                    
                    if path.stat().st_size == 0:
                        continue  # Skip empty files
                    
                    # Quick content check - make sure it's not just whitespace
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            continue
                    
                    notes_to_process.append(note_path)
                    
                except Exception:
                    continue  # Skip files that can't be read
                
                if len(notes_to_process) >= limit:
                    break
        
        print(f"📊 Processing {len(notes_to_process)} notes...")
        
        # Process each note
        results = []
        # Filter out protected test corpus
        try:
            from src.fs_indexer import is_protected_test_content
            notes_to_process = [p for p in notes_to_process if not is_protected_test_content(Path(p))]
        except Exception:
            pass

        for i, note_path in enumerate(notes_to_process, 1):
            print(f"\n[{i}/{len(notes_to_process)}] Processing...")
            result = enricher.enrich_note_properties(note_path, dry_run=dry_run)
            if result:
                results.append(result)
        
        # Summary
        print("\n📈 Enrichment Summary")
        print(f"   • Processed: {len(results)} notes")
        print(f"   • Mode: {'Preview only' if dry_run else 'Changes applied'}")
        
        if results:
            para_types = {}
            for result in results:
                para_type = result.get('classification', {}).get('para_type', 'unknown')
                para_types[para_type] = para_types.get(para_type, 0) + 1
            
            print("   • PARA Distribution:")
            for para_type, count in para_types.items():
                print(f"     - {para_type}: {count}")
        
        if dry_run:
            print("\n💡 To apply changes, run with --apply flag")
        else:
            print("\n✅ Enrichment complete! Graph RAG relationships enhanced.")
            
    except Exception as e:
        print(f"❌ Enrichment failed: {e}")
        raise typer.Exit(1)

@app.command()
def enrich_all(
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview changes without applying them"),
    batch_size: int = typer.Option(50, "--batch-size", help="Number of notes to process in each batch"),
    skip_enriched: bool = typer.Option(True, "--skip-enriched/--force-all", help="Skip notes that already have PARA taxonomy")
):
    """Enrich ALL notes in the vault with PARA taxonomy.
    
    This command processes the entire vault, enriching all notes with PARA classification
    and semantic relationships. Use with caution as this can take a long time for large vaults.
    """
    
    print("🚀 Starting FULL VAULT PARA Taxonomy Enrichment")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY CHANGES'}")
    print(f"Batch size: {batch_size} notes")
    print(f"Skip already enriched: {skip_enriched}")
    print()
    
    # Initialize counters early so they're defined for all code paths
    processed = 0
    
    try:
        # Initialize enricher
        enricher = PARAEnricher()
        
        # Get ALL notes from vault using file discovery
        print("📚 Discovering all notes in vault...")
        from src.fs_indexer import discover_files
        from src.config import settings
        
        all_paths = []
        for path in discover_files(settings.vaults, settings.supported_extensions):
            # Skip empty files
            try:
                if path.stat().st_size > 0:
                    all_paths.append(str(path))
            except Exception:
                continue
        
        print(f"📊 Found {len(all_paths)} notes in vault")
        
        # Filter out already enriched notes if requested
        if skip_enriched:
            print("🔍 Checking for already enriched notes...")
            notes_to_process = []
            already_enriched = 0
            
            for note_path in all_paths:
                try:
                    # Check if note has PARA classification
                    import frontmatter
                    with open(note_path, 'r', encoding='utf-8') as f:
                        post = frontmatter.load(f)
                    
                    if 'para_type' in post.metadata and 'enrichment_version' in post.metadata:
                        already_enriched += 1
                    else:
                        notes_to_process.append(note_path)
                except Exception:
                    # If we can't read it, try to process it
                    notes_to_process.append(note_path)
            
            print(f"   • {already_enriched} notes already enriched (skipping)")
            print(f"   • {len(notes_to_process)} notes need enrichment")
        else:
            notes_to_process = all_paths
            print(f"   • Processing all {len(notes_to_process)} notes (force mode)")
        
        if not notes_to_process:
            print("\n✅ All notes are already enriched!")
            return
        
        # Process in batches
        total_notes = len(notes_to_process)
        processed = 0
        successful = 0
        failed = 0
        para_distribution = {}
        
        print(f"\n📝 Processing {total_notes} notes in batches of {batch_size}...")
        
        for batch_start in range(0, total_notes, batch_size):
            batch_end = min(batch_start + batch_size, total_notes)
            batch = notes_to_process[batch_start:batch_end]
            
            print(f"\n[Batch {batch_start//batch_size + 1}] Processing notes {batch_start + 1}-{batch_end} of {total_notes}")
            
            for i, note_path in enumerate(batch, 1):
                try:
                    # Show progress within batch
                    if i % 10 == 0 or i == len(batch):
                        print(f"   Progress: {i}/{len(batch)} in current batch")
                    
                    result = enricher.enrich_note_properties(note_path, dry_run=dry_run)
                    if result:
                        processed += 1
                        successful += 1
                        
                        # Track PARA distribution
                        para_type = result.get('classification', {}).get('para_type', 'unknown')
                        para_distribution[para_type] = para_distribution.get(para_type, 0) + 1
                    else:
                        processed += 1
                        failed += 1
                except Exception as e:
                    processed += 1
                    failed += 1
                    if i <= 5:  # Only show first few errors to avoid spam
                        print(f"   ⚠️  Error processing {Path(note_path).name}: {e}")
        
        # Final summary
        print("\n" + "="*60)
        print("📈 FULL VAULT Enrichment Summary")
        print("="*60)
        print(f"   • Total processed: {processed} notes")
        print(f"   • Successful: {successful} notes")
        print(f"   • Failed: {failed} notes")
        print(f"   • Mode: {'Preview only' if dry_run else 'Changes applied'}")
        
        if para_distribution:
            print("\n🎯 PARA Distribution:")
            for para_type, count in sorted(para_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / successful * 100) if successful > 0 else 0
                print(f"     • {para_type}: {count} ({percentage:.1f}%)")
        
        if dry_run:
            print("\n💡 To apply changes to the entire vault, run with --apply flag")
            print("⚠️  WARNING: This will modify all notes in your vault!")
        else:
            print("\n✅ Full vault enrichment complete!")
            print("🔄 Remember to reindex with: uv run scripts/reindex.py all")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Enrichment interrupted by user")
        print(f"   • Processed {processed} notes before interruption")
        raise typer.Exit(1)
    except Exception as e:
        print(f"\n❌ Enrichment failed: {e}")
        raise typer.Exit(1)

class Stats(TypedDict):
    total_notes: int
    has_frontmatter: int
    has_para_type: int
    has_tags: int
    para_distribution: Dict[str, int]
    tag_categories: Dict[str, int]

@app.command()
def analyze(
    sample_size: int = typer.Option(20, "--sample", help="Number of notes to analyze"),
):
    """Analyze the current state of PARA taxonomy in the vault."""
    
    print("📊 PARA Taxonomy Analysis")
    
    try:
        # Get sample of notes
        notes = app_state.chroma_store.get_all_notes(limit=sample_size)
        
        stats: Stats = {
            'total_notes': len(notes),
            'has_frontmatter': 0,
            'has_para_type': 0,
            'has_tags': 0,
            'para_distribution': {},
            'tag_categories': {}
        }
        
        for note in notes:
            meta = note.get('meta', {})
            
            # Check for frontmatter
            if meta:
                stats['has_frontmatter'] += 1
                
                # Check for PARA type
                if 'para_type' in meta:
                    stats['has_para_type'] += 1
                    para_type = meta['para_type']
                    stats['para_distribution'][para_type] = stats['para_distribution'].get(para_type, 0) + 1
                
                # Check for tags
                tags = meta.get('tags', [])
                if tags:
                    stats['has_tags'] += 1
                    
                    # Analyze tag structure
                    for tag in tags:
                        if isinstance(tag, str) and '/' in tag:
                            category = tag.split('/')[0]
                            stats['tag_categories'][category] = stats['tag_categories'].get(category, 0) + 1
        
        # Display results
        print(f"\n📈 Analysis Results (sample of {sample_size} notes):")
        total_notes_int = int(stats['total_notes']) if stats['total_notes'] else 0
        hf = int(stats['has_frontmatter'])
        hpt = int(stats['has_para_type'])
        ht = int(stats['has_tags'])
        pct = (hf/total_notes_int*100) if total_notes_int else 0.0
        pct2 = (hpt/total_notes_int*100) if total_notes_int else 0.0
        pct3 = (ht/total_notes_int*100) if total_notes_int else 0.0
        print(f"   • Notes with frontmatter: {hf}/{total_notes_int} ({pct:.1f}%)")
        print(f"   • Notes with PARA type: {hpt}/{total_notes_int} ({pct2:.1f}%)")
        print(f"   • Notes with tags: {ht}/{total_notes_int} ({pct3:.1f}%)")
        
        if stats['para_distribution']:
            print("\n🎯 PARA Distribution:")
            for para_type, count in sorted(stats['para_distribution'].items()):
                print(f"     • {para_type}: {count}")
        
        if stats['tag_categories']:
            print("\n🏷️  Tag Categories:")
            for category, count in sorted(stats['tag_categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     • #{category}: {count}")
        
        # Recommendations
        enrichment_potential = int(stats['total_notes']) - int(stats['has_para_type'])
        print("\n💡 Enrichment Potential:")
        print(f"   • {enrichment_potential} notes could benefit from PARA classification")
        improvement_pct = (enrichment_potential/total_notes_int*100) if total_notes_int else 0.0
        print(f"   • Estimated improvement: {improvement_pct:.1f}% of vault")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
