from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda x: f"OBSIDIAN_RAG_{x.upper()}"
    )
    
    vaults: List[Path] = Field(
        default_factory=lambda: [Path(__file__).parent.parent.parent],
        description="List of Obsidian vault paths to index"
    )
    
    # ChromaDB settings
    chroma_dir: Path = Field(default=Path(".chroma_db"), description="ChromaDB storage directory")
    collection: str = Field(default="obsidian_vault", description="ChromaDB collection name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    
    
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
        default=[".md", ".markdown", ".txt", ".excalidraw", ".base"],
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
    
    # DSPy optimization state settings
    dspy_state_dir: Path = Field(default=Path(".dspy_state"), description="DSPy optimized programs and cache directory")
    dspy_optimize_enabled: bool = Field(default=True, description="Enable DSPy program optimization")
    dspy_auto_mode: Optional[Literal["light", "medium", "heavy"]] = Field(
        default="light", description="MIPROv2 auto mode: light/medium/heavy"
    )
    dspy_eval_dataset_path: Optional[Path] = Field(default=None, description="Path to evaluation dataset")
    dspy_optimization_interval_hours: int = Field(default=168, description="Hours between optimization runs (default: weekly)")
    dspy_max_examples: int = Field(default=50, description="Maximum examples for optimization")
    dspy_bootstrap_demos: int = Field(default=3, description="Max bootstrapped demos for optimization")
    dspy_labeled_demos: int = Field(default=3, description="Max labeled demos for optimization")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load from environment variables
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Vault overrides (support single or multiple paths)
        vault_env = os.getenv("OBSIDIAN_VAULT_PATH") or os.getenv("OBSIDIAN_RAG_VAULTS") or os.getenv("OBSIDIAN_RAG_VAULT_PATH")
        if vault_env:
            # Split by OS path separator and commas
            parts = []
            for chunk in vault_env.split(os.pathsep):
                parts.extend([p for p in chunk.split(',') if p.strip()])
            if parts:
                self.vaults = [Path(p).expanduser().absolute() for p in parts]
        else:
            # Ensure defaults are absolute
            self.vaults = [Path(p).expanduser().absolute() for p in self.vaults]
        
        # Chroma directory override
        chroma_dir_env = os.getenv("OBSIDIAN_RAG_CHROMA_DIR") or os.getenv("CHROMA_DIR")
        if chroma_dir_env:
            self.chroma_dir = Path(chroma_dir_env).expanduser().absolute()
        else:
            self.chroma_dir = Path(self.chroma_dir).expanduser().absolute()
        # Make sure directory exists so persistence is stable
        try:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        
        # DSPy state directory override
        dspy_state_dir_env = os.getenv("OBSIDIAN_RAG_DSPY_STATE_DIR") or os.getenv("DSPY_STATE_DIR")
        if dspy_state_dir_env:
            self.dspy_state_dir = Path(dspy_state_dir_env).expanduser().absolute()
        else:
            self.dspy_state_dir = Path(self.dspy_state_dir).expanduser().absolute()
        # Make sure DSPy state directory exists for persistent optimization
        try:
            self.dspy_state_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        

settings = Settings()
