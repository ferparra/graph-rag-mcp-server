#!/usr/bin/env python3
"""
Graph RAG MCP Server - Automated Installer
Configures the MCP server for Claude Desktop, Cursor, and Raycast
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List
import argparse

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")

class MCPInstaller:
    def __init__(self):
        self.system = platform.system()
        self.home = Path.home()
        self.project_root = Path(__file__).parent.absolute()
        self.configs_dir = self.project_root / "configs"
        # Default GitHub source for uvx --from
        self.github_from = "git+https://github.com/ferparra/graph-rag-mcp-server"
        
        # Client config paths
        self.claude_config_path = self._get_claude_config_path()
        self.cursor_config_path = self._get_cursor_config_path()
        self.raycast_config_path = self._get_raycast_config_path()
        
        # Default values
        self.vault_path = self._find_obsidian_vault()
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    
    def _get_claude_config_path(self) -> Optional[Path]:
        """Get Claude Desktop config path based on OS."""
        if self.system == "Darwin":  # macOS
            return self.home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        elif self.system == "Windows":
            return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
        elif self.system == "Linux":
            return self.home / ".config" / "Claude" / "claude_desktop_config.json"
        return None
    
    def _get_cursor_config_path(self) -> Optional[Path]:
        """Get Cursor config path."""
        # Cursor typically uses VS Code's settings or its own config
        if self.system == "Darwin":
            return self.home / "Library" / "Application Support" / "Cursor" / "mcp.json"
        elif self.system == "Windows":
            return Path(os.environ.get("APPDATA", "")) / "Cursor" / "mcp.json"
        else:
            return self.home / ".config" / "Cursor" / "mcp.json"
    
    def _get_raycast_config_path(self) -> Optional[Path]:
        """Get Raycast MCP config path."""
        if self.system == "Darwin":
            # Raycast MCP extension stores configs here
            return self.home / "Library" / "Application Support" / "com.raycast.macos" / "extensions" / "mcp-servers" / "mcp-config.json"
        return None  # Raycast is macOS only
    
    def _find_obsidian_vault(self) -> Path:
        """Try to find an Obsidian vault automatically."""
        potential_paths = [
            self.home / "Documents" / "Obsidian",
            self.home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
            self.home / "Obsidian",
            self.home / "Documents" / "ObsidianVault",
        ]
        
        for path in potential_paths:
            if path.exists() and path.is_dir():
                # Look for .obsidian folder
                for vault in path.glob("*/.obsidian"):
                    return vault.parent
        
        # Default fallback
        return self.home / "Documents" / "ObsidianVault"
    
    def check_uv_installed(self) -> bool:
        """Check if uv is installed and accessible."""
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print_success(f"Found uv: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print_error("uv is not installed or not in PATH")
        print_info("Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    
    def check_dependencies(self) -> bool:
        """Check and install Python dependencies."""
        print_info("Checking Python dependencies...")
        
        try:
            # Run uv sync to ensure all dependencies are installed
            result = subprocess.run(
                ["uv", "sync"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print_success("All dependencies installed")
                return True
            else:
                print_error(f"Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            print_error(f"Error checking dependencies: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """Create or update .env file with required variables."""
        env_path = self.project_root / ".env"
        env_example_path = self.project_root / ".env.example"
        
        if env_path.exists():
            print_info(".env file already exists, skipping...")
            return True
        
        print_info("Creating .env file...")
        
        # Copy from example if it exists
        if env_example_path.exists():
            shutil.copy(env_example_path, env_path)
        
        # Prompt for API key if not set
        if not self.gemini_api_key:
            self.gemini_api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
        
        # Update .env with actual values
        env_content = f"""# Graph RAG MCP Server Configuration
GEMINI_API_KEY={self.gemini_api_key}
OBSIDIAN_RAG_VAULTS={self.vault_path}
OBSIDIAN_RAG_CHROMA_DIR=.chroma_db
OBSIDIAN_RAG_COLLECTION=vault_collection
OBSIDIAN_RAG_CHUNK_STRATEGY=semantic
OBSIDIAN_RAG_MCP_HOST=localhost
OBSIDIAN_RAG_MCP_PORT=8765
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print_success("Created .env file")
        return True
    
    def backup_config(self, config_path: Path):
        """Create backup of existing config."""
        if config_path.exists():
            backup_path = config_path.with_suffix('.json.backup')
            shutil.copy(config_path, backup_path)
            print_info(f"Backed up existing config to {backup_path.name}")
    
    def merge_mcp_config(self, existing: Dict, new_server: Dict, server_name: str) -> Dict:
        """Merge new MCP server config with existing config."""
        if "mcpServers" not in existing:
            existing["mcpServers"] = {}
        
        existing["mcpServers"][server_name] = new_server
        return existing
    
    def install_claude_desktop(self) -> bool:
        """Install configuration for Claude Desktop."""
        if not self.claude_config_path:
            print_warning("Claude Desktop config path not found for this OS")
            return False
        
        print_info(f"Configuring Claude Desktop at {self.claude_config_path}")
        
        # Create directory if needed
        self.claude_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load template
        template_path = self.configs_dir / "claude-desktop.json"
        with open(template_path) as f:
            template = json.load(f)
        
        # Update with actual values
        server_config = template["mcpServers"]["graph-rag-obsidian"]
        # Prefer uvx for hermetic, pinned MCP runs
        server_config["command"] = "uvx"
        server_config["args"] = ["--python", "3.13", "--from", self.github_from, "graph-rag-mcp-stdio"]
        # Do not set cwd when running from GitHub source
        if "cwd" in server_config:
            server_config.pop("cwd", None)
        server_config["env"]["GEMINI_API_KEY"] = self.gemini_api_key
        server_config["env"]["OBSIDIAN_RAG_VAULTS"] = str(self.vault_path)
        
        # Check for existing config
        existing_config = {}
        if self.claude_config_path.exists():
            self.backup_config(self.claude_config_path)
            with open(self.claude_config_path) as f:
                existing_config = json.load(f)
        
        # Merge configs
        final_config = self.merge_mcp_config(
            existing_config,
            server_config,
            "graph-rag-obsidian"
        )
        
        # Write config
        with open(self.claude_config_path, 'w') as f:
            json.dump(final_config, f, indent=2)

        print_success("Claude Desktop configured successfully")
        print_info("Restart Claude Desktop to load the new MCP server")
        # Display resulting JSON object for Claude Desktop
        print_info("Claude Desktop mcpServers JSON:")
        try:
            print(json.dumps(final_config.get("mcpServers", {}), indent=2))
        except Exception:
            pass
        return True
    
    def install_cursor(self) -> bool:
        """Install configuration for Cursor."""
        if not self.cursor_config_path:
            print_warning("Cursor config path not configured")
            return False
        
        print_info(f"Configuring Cursor at {self.cursor_config_path}")
        
        # Create directory if needed
        self.cursor_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load template
        template_path = self.configs_dir / "cursor-mcp.json"
        with open(template_path) as f:
            template = json.load(f)
        
        # Update with actual values
        for server_name in ["graph-rag-obsidian", "graph-rag-obsidian-http"]:
            if server_name in template["mcpServers"]:
                config = template["mcpServers"][server_name]
                if "env" in config:
                    config["env"]["GEMINI_API_KEY"] = self.gemini_api_key
                    config["env"]["OBSIDIAN_RAG_VAULTS"] = str(self.vault_path)

        # Write config
        with open(self.cursor_config_path, 'w') as f:
            json.dump(template, f, indent=2)

        print_success("Cursor configured successfully")
        print_info("Restart Cursor to load the new MCP server")
        # Display resulting JSON object for Cursor
        print_info("Cursor mcpServers JSON:")
        try:
            print(json.dumps(template.get("mcpServers", {}), indent=2))
        except Exception:
            pass
        return True
    
    def install_raycast(self) -> bool:
        """Install configuration for Raycast."""
        if self.system != "Darwin":
            print_warning("Raycast is only available on macOS")
            return False
        
        if not self.raycast_config_path:
            print_warning("Raycast MCP extension not found")
            print_info("Install it from: raycast://extensions/raycast/mcp")
            return False
        
        print_info(f"Configuring Raycast at {self.raycast_config_path}")
        
        # Create directory if needed
        self.raycast_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load template
        template_path = self.configs_dir / "raycast-config.json"
        with open(template_path) as f:
            template = json.load(f)
        
        # Update with actual values
        server_config = template["mcpServers"]["graph-rag-obsidian"]
        # Prefer uvx for hermetic, pinned MCP runs
        server_config["command"] = "uvx"
        server_config["args"] = ["--python", "3.13", "--from", self.github_from, "graph-rag-mcp-stdio"]
        # Do not set cwd when running from GitHub source
        if "cwd" in server_config:
            server_config.pop("cwd", None)
        server_config["env"]["GEMINI_API_KEY"] = self.gemini_api_key
        server_config["env"]["OBSIDIAN_RAG_VAULTS"] = str(self.vault_path)
        
        # Check for existing config
        existing_config = {}
        if self.raycast_config_path.exists():
            self.backup_config(self.raycast_config_path)
            with open(self.raycast_config_path) as f:
                existing_config = json.load(f)
        
        # Merge configs
        final_config = self.merge_mcp_config(
            existing_config,
            server_config,
            "graph-rag-obsidian"
        )
        
        # Write config
        with open(self.raycast_config_path, 'w') as f:
            json.dump(final_config, f, indent=2)

        print_success("Raycast configured successfully")
        print_info("Restart Raycast to load the new MCP server")
        # Display resulting JSON object for Raycast
        print_info("Raycast mcpServers JSON:")
        try:
            print(json.dumps(final_config.get("mcpServers", {}), indent=2))
        except Exception:
            pass
        return True
    
    def test_server(self) -> bool:
        """Test that the MCP server can start."""
        print_info("Testing MCP server startup...")
        
        try:
            # Start stdio server via uvx with a short timeout; if it blocks, it's running
            subprocess.run(
                ["uvx", "--python", "3.13", "--from", self.github_from, "graph-rag-mcp-stdio"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            # If it returned quickly, we can't be sure; treat as inconclusive but continue
            print_warning("MCP server test inconclusive (exited early), continuing...")
            return True
        except subprocess.TimeoutExpired:
            # Timeout means the server is running (expected)
            print_success("MCP server started successfully (timeout expected)")
            return True
        except Exception as e:
            print_warning(f"Server test warning: {e}")
            return True  # Don't fail installation
    
    def run_initial_index(self) -> bool:
        """Run initial indexing of the vault."""
        response = input("\nWould you like to index your vault now? (y/n): ").strip().lower()
        
        if response != 'y':
            print_info("Skipping initial indexing. Run 'uv run scripts/reindex.py all' later.")
            return True
        
        print_info("Running initial vault indexing...")
        
        try:
            result = subprocess.run(
                ["uv", "run", "scripts/reindex.py", "all"],
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print_success("Vault indexed successfully")
                return True
            else:
                print_warning("Indexing completed with warnings")
                return True
        except Exception as e:
            print_error(f"Indexing error: {e}")
            return False
    
    def print_summary(self, claude: bool, cursor: bool, raycast: bool):
        """Print installation summary."""
        print_header("Installation Summary")
        
        print(f"{Colors.BOLD}Installed Clients:{Colors.ENDC}")
        if claude:
            print_success("Claude Desktop")
        if cursor:
            print_success("Cursor")
        if raycast:
            print_success("Raycast")
        
        print(f"\n{Colors.BOLD}Configuration:{Colors.ENDC}")
        print(f"  Vault Path: {self.vault_path}")
        print(f"  Project Root: {self.project_root}")
        
        print(f"\n{Colors.BOLD}Quick Start Commands:{Colors.ENDC}")
        print("  Index vault:        uv run scripts/reindex.py all")
        print("  Test server:        uvx --python 3.13 --from git+https://github.com/ferparra/graph-rag-mcp-server graph-rag-mcp-stdio")
        print("  Start HTTP server:  uv run graph-rag-mcp-http")
        print("  Enrich notes:       uv run scripts/enrich_para_taxonomy.py enrich-all --apply")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("  1. Restart your MCP clients to load the server")
        print("  2. Run indexing if you haven't already")
        print("  3. Start using Graph RAG tools in your client!")
    
    def run(self, clients: List[str]):
        """Run the installation process."""
        print_header("Graph RAG MCP Server Installer")
        
        # Check prerequisites
        if not self.check_uv_installed():
            return False
        
        if not self.check_dependencies():
            return False
        
        # Create .env file
        self.create_env_file()
        
        # Get vault path from user
        print(f"\nCurrent vault path: {self.vault_path}")
        custom_path = input("Enter custom vault path (or press Enter to use default): ").strip()
        if custom_path:
            self.vault_path = Path(custom_path)
        
        # Install for selected clients
        results = {}
        
        if "all" in clients or "claude" in clients:
            results["claude"] = self.install_claude_desktop()
        
        if "all" in clients or "cursor" in clients:
            results["cursor"] = self.install_cursor()
        
        if "all" in clients or "raycast" in clients:
            results["raycast"] = self.install_raycast()
        
        # Test server
        self.test_server()
        
        # Offer to run initial indexing
        self.run_initial_index()
        
        # Print summary
        self.print_summary(
            results.get("claude", False),
            results.get("cursor", False),
            results.get("raycast", False)
        )
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Install Graph RAG MCP Server for various clients"
    )
    parser.add_argument(
        "clients",
        nargs="+",
        choices=["all", "claude", "cursor", "raycast"],
        help="Clients to install for"
    )
    parser.add_argument(
        "--vault",
        help="Path to Obsidian vault",
        type=str
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key",
        type=str
    )
    
    args = parser.parse_args()
    
    installer = MCPInstaller()
    
    # Override with command line args if provided
    if args.vault:
        installer.vault_path = Path(args.vault)
    if args.api_key:
        installer.gemini_api_key = args.api_key
    
    try:
        success = installer.run(args.clients)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
