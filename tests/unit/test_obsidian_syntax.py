"""Tests for Obsidian Bases syntax compliance helpers."""

import yaml
from src.base_parser import ObsidianBaseFile, ObsidianView, ObsidianBase, FilterGroup


def test_obsidian_base_minimal_yaml_roundtrip():
    obs = ObsidianBaseFile(
        schema_="vault://schemas/obsidian/bases-2025-09.schema.json",
        views=[ObsidianView(type="table", name="My table", limit=10)],
    )
    y = ObsidianBase.to_yaml(obs)
    assert "views:" in y and "type: table" in y
    # Ensure safe-loadable and structure preserved
    data = yaml.safe_load(y)
    assert data["views"][0]["type"] == "table"
    assert data["views"][0]["name"] == "My table"
    assert data["views"][0]["limit"] == 10


def test_obsidian_filters_structure_and_strings():
    # Mixed filter: and of strings
    fg = FilterGroup.model_validate({"and": ["status != \"done\"", "price > 2.1"]})
    obs = ObsidianBaseFile(views=[ObsidianView(type="cards", name="Cards")], filters=fg)
    y = ObsidianBase.to_yaml(obs)
    # Verify YAML structure contains 'filters:\n  and:'
    assert "filters:" in y and "and:" in y
    # Roundtrip parse
    parsed = ObsidianBase.parse_yaml(y)
    assert isinstance(parsed, ObsidianBaseFile)

