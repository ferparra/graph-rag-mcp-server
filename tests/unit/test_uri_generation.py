#!/usr/bin/env python3
"""
Unit tests for Obsidian URI generation logic.
Tests the URI construction and anchor generation without external dependencies.
"""

from typing import Dict, Any


def generate_chunk_uri(chunk_metadata: Dict[str, Any], vault_name: str = "TestVault") -> str:
    """Generate Obsidian URI for a specific chunk"""
    note_path = chunk_metadata.get('path', '')
    header_text = chunk_metadata.get('header_text', '')
    
    # Remove vault path prefix to get relative path
    relative_path = note_path
    if '/' in note_path:
        relative_path = note_path.split('/')[-1]
    
    # Remove .md extension
    if relative_path.endswith('.md'):
        relative_path = relative_path[:-3]
    
    base_uri = f"obsidian://open?vault={vault_name}&file={relative_path}"
    
    # Add header anchor if available
    if header_text:
        # Convert header to URL-safe anchor
        anchor = header_text.lower().replace(' ', '%20').replace('#', '')
        base_uri += f"#{anchor}"
    
    return base_uri


def test_basic_uri_generation():
    """Test basic URI generation for different note types"""
    
    test_cases = [
        {
            "metadata": {
                'path': 'tests/fixtures/content/planets/Earth.md',
                'header_text': 'Earth',
                'title': 'Earth'
            },
            "vault": "TestVault",
            "expected_file": "Earth",
            "expected_anchor": "earth"
        },
        {
            "metadata": {
                'path': 'tests/fixtures/content/planets/Mars.md',
                'header_text': 'Mars',
                'title': 'Mars'
            },
            "vault": "MyVault",
            "expected_file": "Mars",
            "expected_anchor": "mars"
        },
        {
            "metadata": {
                'path': 'tests/fixtures/content/links/Link Map.md',
                'header_text': 'Link Map',
                'title': 'Link Map'
            },
            "vault": "TestVault",
            "expected_file": "Link Map",
            "expected_anchor": "link%20map"
        }
    ]
    
    passed = 0
    for i, test in enumerate(test_cases):
        uri = generate_chunk_uri(test["metadata"], test["vault"])
        
        print(f"Test {i+1}: {test['metadata']['title']}")
        print(f"  Generated: {uri}")
        
        # Check key components
        checks = [
            "obsidian://open" in uri,
            f"vault={test['vault']}" in uri,
            f"file={test['expected_file']}" in uri,
            f"#{test['expected_anchor']}" in uri
        ]
        
        check_names = ["protocol", "vault", "file", "anchor"]
        
        all_passed = True
        for j, (check, name) in enumerate(zip(checks, check_names)):
            if check:
                print(f"    ✓ {name} component correct")
            else:
                print(f"    ❌ {name} component incorrect")
                all_passed = False
        
        if all_passed:
            print("  ✓ URI generation correct")
            passed += 1
        else:
            print("  ❌ URI generation incorrect")
        print()
    
    print(f"Basic URI generation: {passed}/{len(test_cases)} tests passed")
    assert passed == len(test_cases)


def test_edge_cases():
    """Test edge cases for URI generation"""
    
    edge_cases = [
        {
            "name": "No header text",
            "metadata": {'path': 'test/note.md', 'title': 'Test Note'},
            "vault": "Vault",
            "should_have_anchor": False
        },
        {
            "name": "Complex header with special chars",
            "metadata": {'path': 'notes/complex.md', 'header_text': 'Complex Header #123!'},
            "vault": "Vault",
            "should_have_anchor": True,
            "expected_anchor": "complex%20header%20123!"
        },
        {
            "name": "Empty path",
            "metadata": {'path': '', 'header_text': 'Header'},
            "vault": "Vault",
            "should_have_anchor": True
        },
        {
            "name": "Path without extension",
            "metadata": {'path': 'notes/without_ext', 'header_text': 'Header'},
            "vault": "Vault",
            "should_have_anchor": True
        },
        {
            "name": "Very long vault name",
            "metadata": {'path': 'test.md', 'header_text': 'Test'},
            "vault": "Very Long Vault Name With Spaces",
            "should_have_anchor": True
        }
    ]
    
    passed = 0
    for case in edge_cases:
        uri = generate_chunk_uri(case["metadata"], case["vault"])
        
        print(f"Edge case: {case['name']}")
        print(f"  Generated: {uri}")
        
        # Basic checks
        has_protocol = "obsidian://open" in uri
        has_vault = f"vault={case['vault']}" in uri
        has_anchor = "#" in uri
        
        success = True
        
        if not has_protocol:
            print("  ❌ Missing protocol")
            success = False
        else:
            print("  ✓ Protocol present")
        
        if not has_vault:
            print("  ❌ Missing vault parameter")
            success = False
        else:
            print("  ✓ Vault parameter present")
        
        if case["should_have_anchor"]:
            if not has_anchor:
                print("  ❌ Expected anchor but none found")
                success = False
            else:
                print("  ✓ Anchor present as expected")
                
                # Check specific anchor if provided
                if "expected_anchor" in case:
                    if case["expected_anchor"] in uri:
                        print("  ✓ Anchor content correct")
                    else:
                        print(f"  ❌ Expected anchor '{case['expected_anchor']}' not found")
                        success = False
        else:
            if has_anchor:
                print("  ❌ Unexpected anchor found")
                success = False
            else:
                print("  ✓ No anchor as expected")
        
        if success:
            passed += 1
        print()
    
    print(f"Edge cases: {passed}/{len(edge_cases)} tests passed")
    assert passed == len(edge_cases)


def test_anchor_encoding():
    """Test URL encoding of header anchors"""
    
    encoding_cases = [
        ("Simple Header", "simple%20header"),
        ("Header With Spaces", "header%20with%20spaces"),
        ("Header#With#Hashes", "headerwithhashes"),  # Hashes removed
        ("CamelCase", "camelcase"),  # Converted to lowercase
        ("Numbers 123", "numbers%20123"),
        ("Special!@#$%^&*()", "special!@$%^&*()"),  # Hashes removed, others kept
    ]
    
    passed = 0
    for header_text, expected_anchor in encoding_cases:
        metadata = {
            'path': 'test.md',
            'header_text': header_text
        }
        
        uri = generate_chunk_uri(metadata, "TestVault")
        
        print(f"Header: '{header_text}'")
        print(f"  Expected anchor: {expected_anchor}")
        print(f"  Generated URI: {uri}")
        
        if f"#{expected_anchor}" in uri:
            print("  ✓ Anchor encoding correct")
            passed += 1
        else:
            print("  ❌ Anchor encoding incorrect")
        print()
    
    print(f"Anchor encoding: {passed}/{len(encoding_cases)} tests passed")
    assert passed == len(encoding_cases)


def test_file_path_handling():
    """Test different file path formats"""
    
    path_cases = [
        ("simple.md", "simple"),
        ("folder/file.md", "file"),
        ("deep/nested/path/file.md", "file"),
        ("file_without_extension", "file_without_extension"),
        ("file.with.dots.md", "file.with.dots"),
        ("", ""),  # Empty path
    ]
    
    passed = 0
    for input_path, expected_file in path_cases:
        metadata = {
            'path': input_path,
            'header_text': 'Test'
        }
        
        uri = generate_chunk_uri(metadata, "TestVault")
        
        print(f"Path: '{input_path}'")
        print(f"  Expected file: '{expected_file}'")
        print(f"  Generated URI: {uri}")
        
        if expected_file:
            if f"file={expected_file}" in uri:
                print("  ✓ File path handling correct")
                passed += 1
            else:
                print("  ❌ File path handling incorrect")
        else:
            # Empty file case - just check it doesn't crash
            if "file=" in uri:
                print("  ✓ Empty path handled gracefully")
                passed += 1
            else:
                print("  ❌ Empty path not handled")
        print()
    
    print(f"File path handling: {passed}/{len(path_cases)} tests passed")
    assert passed == len(path_cases)


if __name__ == "__main__":
    print("🧪 Testing URI Generation")
    print("=" * 50)
    
    basic_test = test_basic_uri_generation()
    edge_test = test_edge_cases()
    encoding_test = test_anchor_encoding()
    path_test = test_file_path_handling()
    
    if basic_test and edge_test and encoding_test and path_test:
        print("\n✅ All URI generation tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
