from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    vaults: List[Path] = Field(
        default_factory=lambda: [Path(__file__).parent.parent.parent],
        description="List of Obsidian vault paths to index"
    )
    
    # ChromaDB settings
    chroma_dir: Path = Field(default=Path(".chroma_db"), description="ChromaDB storage directory")
    collection: str = Field(default="obsidian_vault", description="ChromaDB collection name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    
    # RDF/Oxigraph settings
    rdf_db_path: Path = Field(default=Path(".vault_graph.db"), description="Base path for RDF store (Oxigraph will create a directory)")
    rdf_store_identifier: str = Field(default="obsidian_vault_graph", description="RDF store identifier")
    
    # Gemini settings (using google-genai SDK)
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key")
    
    # Chunking settings
    max_chars: int = Field(default=1800, description="Maximum characters per chunk")
    overlap: int = Field(default=200, description="Character overlap between chunks")
    
    # MCP server settings
    mcp_port: int = Field(default=8765, description="MCP server port for HTTP transport")
    mcp_host: str = Field(default="localhost", description="MCP server host")
    
    # File processing settings
    supported_extensions: List[str] = Field(
        default=[".md", ".markdown", ".txt", ".excalidraw"],
        description="File extensions to process"
    )
    
    # Archive settings
    archive_folder: str = Field(default="Archive", description="Folder name for archived notes")
    
    # Semantic chunking settings
    chunk_strategy: str = Field(default="semantic", description="Chunking strategy: 'semantic' or 'character'")
    semantic_min_chunk_size: int = Field(default=100, description="Minimum size for semantic chunks")
    semantic_max_chunk_size: int = Field(default=3000, description="Maximum size for semantic chunks")
    semantic_merge_threshold: int = Field(default=200, description="Merge chunks smaller than this threshold")
    semantic_include_context: bool = Field(default=True, description="Include parent headers as context in chunks")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load from environment variables
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Override RDF settings with environment variables if available
        rdf_db_path_env = os.getenv("RDF_DB_PATH")
        if rdf_db_path_env:
            self.rdf_db_path = Path(rdf_db_path_env)
        
        rdf_store_id_env = os.getenv("RDF_STORE_IDENTIFIER")
        if rdf_store_id_env:
            self.rdf_store_identifier = rdf_store_id_env
    
    class Config:
        env_prefix = "OBSIDIAN_RAG_"

settings = Settings()
