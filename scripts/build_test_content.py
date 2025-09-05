from __future__ import annotations
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKING_DIR = ROOT / "rag_mcp_server_test_content"
BASELINE_DIR = ROOT / "rag_mcp_server_test_content_baseline"

FILES: dict[str, str] = {
    "README.md": """---
title: Test Corpus Main
tags: [test/suite, corpus]
---

# Graph RAG MCP Test Corpus

This folder contains a tiny, deterministic corpus for evaluating MCP tools.

- Use cases: vector search, graph neighbors, backlinks, tags, CRUD.
- Each note includes distinct phrases to avoid ambiguity across embedding models.

See `scripts/run_mcp_evals.py` for automated checks.
""",
    "planets/Earth.md": """---
title: Earth
tags: [test/suite, topic/planets]
---

# Earth

The Earth verification code is 93E8A4.

Deterministic anchor text: CONST_TOKEN_ALPHA_earth_93e8a4.

Facts:
- Third planet from the Sun.
- Connected to [[Mars]] via the Link Map.
""",
    "planets/Mars.md": """---
title: Mars
tags: [test/suite, topic/planets]
---

# Mars

The Mars verification code is 77C2D1.

Deterministic anchor text: CONST_TOKEN_ALPHA_mars_77c2d1.

Facts:
- Fourth planet from the Sun.
- Connected to [[Earth]] via the Link Map.
""",
    "links/Link Map.md": """---
title: Link Map
tags: [test/suite, links]
---

# Link Map

This note intentionally links the planets to exercise graph traversal and backlinks.

Links:
- [[Earth]]
- [[Mars]]

Deterministic anchor text: CONST_LINK_HUB_0001
""",
}


def write_baseline() -> None:
    """Write the immutable baseline corpus to BASELINE_DIR."""
    for rel, content in FILES.items():
        p = BASELINE_DIR / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.strip() + "\n", encoding="utf-8")


def reset_working_from_baseline() -> None:
    """Reset the working directory from the immutable baseline copy."""
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    shutil.copytree(BASELINE_DIR, WORKING_DIR)


def write_files() -> None:
    """Backward-compatible helper: writes baseline and resets working from it."""
    write_baseline()
    reset_working_from_baseline()


if __name__ == "__main__":
    write_files()
    print(f"Baseline: {BASELINE_DIR}")
    print(f"Working copy: {WORKING_DIR}")
