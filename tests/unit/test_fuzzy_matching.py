#!/usr/bin/env python3
"""
Unit tests for fuzzy tag matching algorithms.
Tests the string similarity and tag relevance scoring without external dependencies.
"""

from typing import List, TypedDict


def get_bigrams(s: str) -> set:
    """Get character bigrams for Jaccard similarity"""
    return set(s[i:i+2] for i in range(len(s)-1))


def fuzzy_string_match(s1: str, s2: str) -> float:
    """Jaccard similarity on character bigrams"""
    if not s1 or not s2:
        return 0.0
        
    bigrams1 = get_bigrams(s1)
    bigrams2 = get_bigrams(s2)
    
    if not bigrams1 and not bigrams2:
        return 1.0 if s1 == s2 else 0.0
    if not bigrams1 or not bigrams2:
        return 0.0
        
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Boost score for similar length strings
    length_similarity = 1.0 - abs(len(s1) - len(s2)) / max(len(s1), len(s2))
    return (jaccard * 0.7 + length_similarity * 0.3) * 0.5


def calculate_tag_relevance_score(query_entities: List[str], note_tags: List[str]) -> float:
    """Calculate relevance score between query entities and note tags"""
    if not query_entities or not note_tags:
        return 0.0
        
    total_score = 0.0
    max_possible_score = len(query_entities)
    
    for entity in query_entities:
        entity_lower = entity.lower()
        best_match_score = 0.0
        
        for tag in note_tags:
            tag_lower = tag.lower()
            
            # Exact match (highest score)
            if entity_lower == tag_lower:
                best_match_score = max(best_match_score, 1.0)
                continue
            
            # Hierarchical tag matching
            if '/' in tag_lower:
                tag_parts = tag_lower.split('/')
                if entity_lower in tag_parts:
                    best_match_score = max(best_match_score, 0.9)
                    continue
                
                # Partial hierarchical match
                for part in tag_parts:
                    if entity_lower in part or part in entity_lower:
                        best_match_score = max(best_match_score, 0.7)
            
            # Substring matching
            if entity_lower in tag_lower:
                best_match_score = max(best_match_score, 0.8)
            elif tag_lower in entity_lower:
                best_match_score = max(best_match_score, 0.6)
            
            # Fuzzy matching
            score = fuzzy_string_match(entity_lower, tag_lower)
            best_match_score = max(best_match_score, score)
        
        total_score += best_match_score
    
    return total_score / max_possible_score


def test_fuzzy_string_matching():
    """Test fuzzy string matching algorithm"""
    
    test_cases = [
        ("health", "health", 0.5),  # Exact match scaled by 0.5
        ("planet", "planets", 0.2),  # Similar but not exact
        ("test", "suite", 0.0),  # No similarity
        ("topic", "topics", 0.2),  # Plural vs singular
        ("", "", 0.0),  # Empty strings
        ("a", "ab", 0.0),  # Very short strings
    ]
    
    passed = 0
    for s1, s2, expected_min in test_cases:
        score = fuzzy_string_match(s1, s2)
        
        print(f"'{s1}' vs '{s2}': {score:.3f} (expected ≥{expected_min:.3f})")
        
        if score >= expected_min:
            print("  ✓ Fuzzy matching working correctly")
            passed += 1
        else:
            print("  ❌ Fuzzy matching score too low")
        print()
    
    print(f"Fuzzy string matching: {passed}/{len(test_cases)} tests passed")
    assert passed == len(test_cases)


def test_tag_relevance_scoring():
    """Test tag relevance scoring with hierarchical tags"""
    
    # Test with planet test data tags
    class TagScoreCase(TypedDict):
        query_entities: List[str]
        note_tags: List[str]
        expected_score: float

    test_cases: List[TagScoreCase] = [
        {
            "query_entities": ["planets"],
            "note_tags": ["test/suite", "topic/planets"],
            "expected_score": 0.9  # Should match "topic/planets" hierarchically
        },
        {
            "query_entities": ["test"],
            "note_tags": ["test/suite", "topic/planets"],
            "expected_score": 0.9  # Should match "test/suite" hierarchically
        },
        {
            "query_entities": ["suite"],
            "note_tags": ["test/suite", "links"],
            "expected_score": 0.9  # Should match "test/suite" hierarchically
        },
        {
            "query_entities": ["links"],
            "note_tags": ["test/suite", "links"],
            "expected_score": 1.0  # Exact match
        },
        {
            "query_entities": ["planet"],  # Singular vs plural
            "note_tags": ["topic/planets"],
            "expected_score": 0.2  # Should get some fuzzy match score
        },
        {
            "query_entities": ["health", "fitness"],
            "note_tags": ["para/area/health", "lifestyle/fitness"],
            "expected_score": 0.9  # Both should match hierarchically
        }
    ]
    
    passed = 0
    for i, test in enumerate(test_cases):
        score = calculate_tag_relevance_score(test["query_entities"], test["note_tags"])
        expected = test["expected_score"]
        
        print(f"Test {i+1}: {test['query_entities']} vs {test['note_tags']}")
        print(f"  Score: {score:.3f}, Expected: ≥{expected:.3f}")
        
        if score >= expected * 0.8:  # Allow some tolerance
            print("  ✓ Tag matching working correctly")
            passed += 1
        else:
            print("  ❌ Tag matching score too low")
        print()
    
    print(f"Tag relevance scoring: {passed}/{len(test_cases)} tests passed")
    assert passed == len(test_cases)


def test_hierarchical_tag_matching():
    """Test specific hierarchical tag patterns"""
    
    hierarchical_cases = [
        # (query, tag, expected_min_score)
        ("health", "para/area/health", 0.9),
        ("area", "para/area/health", 0.9), 
        ("para", "para/area/health", 0.9),
        ("fitness", "lifestyle/fitness/goals", 0.9),
        ("goals", "fitness/goals", 0.9),
        ("unknown", "para/area/health", 0.0),
    ]
    
    passed = 0
    for query, tag, expected_min in hierarchical_cases:
        score = calculate_tag_relevance_score([query], [tag])
        
        print(f"'{query}' vs '{tag}': {score:.3f} (expected ≥{expected_min:.3f})")
        
        if score >= expected_min:
            print("  ✓ Hierarchical matching working")
            passed += 1
        else:
            print("  ❌ Hierarchical matching failed")
        print()
    
    print(f"Hierarchical matching: {passed}/{len(hierarchical_cases)} tests passed")
    assert passed == len(hierarchical_cases)


if __name__ == "__main__":
    print("🧪 Testing Fuzzy Tag Matching")
    print("=" * 50)
    
    string_test = test_fuzzy_string_matching()
    relevance_test = test_tag_relevance_scoring()
    hierarchical_test = test_hierarchical_tag_matching()
    
    if string_test and relevance_test and hierarchical_test:
        print("\n✅ All fuzzy matching tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
