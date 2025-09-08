#!/usr/bin/env python3
"""
Unit tests for graph relationship weighting logic.
Tests the relationship scoring and depth decay algorithms without external dependencies.
"""


def get_relationship_weight(relationship: str, depth: int) -> float:
    """Calculate weight for different relationship types, adjusted by depth"""
    base_weights = {
        "sequential_next": 0.8,
        "sequential_prev": 0.8,
        "parent": 0.9,
        "child": 0.7,
        "sibling": 0.6,
        "content_link": 0.9
    }
    
    weight = base_weights.get(relationship, 0.5)
    depth_factor = 0.8 ** depth
    return weight * depth_factor


def test_base_relationship_weights():
    """Test base weights for different relationship types"""
    
    expected_weights = {
        "content_link": 0.9,    # Highest - direct content links
        "parent": 0.9,          # High - parent sections  
        "sequential_next": 0.8, # High - document flow
        "sequential_prev": 0.8, # High - document flow
        "child": 0.7,           # Medium - subsections
        "sibling": 0.6,         # Lower - peer sections
        "unknown": 0.5,         # Default for unknown types
    }
    
    passed = 0
    for relationship, expected in expected_weights.items():
        weight = get_relationship_weight(relationship, 0)  # Depth 0 = no decay
        
        print(f"{relationship}: {weight:.3f} (expected: {expected:.3f})")
        
        if abs(weight - expected) < 0.01:
            print("  ✓ Base weight correct")
            passed += 1
        else:
            print("  ❌ Base weight incorrect")
        print()
    
    print(f"Base weights: {passed}/{len(expected_weights)} tests passed")
    assert passed == len(expected_weights)


def test_depth_decay():
    """Test depth-based weight decay"""
    
    # Test relationship weights at different depths
    test_cases = [
        ("content_link", 0, 0.9),      # No decay at depth 0
        ("content_link", 1, 0.72),     # 0.9 * 0.8 = 0.72
        ("content_link", 2, 0.576),    # 0.9 * 0.8^2 = 0.576
        ("parent", 1, 0.72),           # 0.9 * 0.8 = 0.72
        ("sequential_next", 1, 0.64),  # 0.8 * 0.8 = 0.64
        ("child", 1, 0.56),            # 0.7 * 0.8 = 0.56
        ("sibling", 2, 0.384),         # 0.6 * 0.8^2 = 0.384
    ]
    
    passed = 0
    for relationship, depth, expected in test_cases:
        weight = get_relationship_weight(relationship, depth)
        
        print(f"{relationship} at depth {depth}: {weight:.3f} (expected: {expected:.3f})")
        
        if abs(weight - expected) < 0.01:  # Small tolerance for floating point
            print("  ✓ Weight calculation correct")
            passed += 1
        else:
            print("  ❌ Weight calculation incorrect")
        print()
    
    print(f"Depth decay: {passed}/{len(test_cases)} tests passed")
    assert passed == len(test_cases)


def test_weight_ordering():
    """Test that relationship weights maintain expected ordering"""
    
    # At the same depth, these should be ordered by importance
    relationships_by_priority = [
        "content_link",    # 0.9 - Most important
        "parent",          # 0.9 - Most important  
        "sequential_next", # 0.8
        "sequential_prev", # 0.8
        "child",           # 0.7
        "sibling",         # 0.6 - Least important
    ]
    
    depth = 1  # Test at depth 1 to include decay factor
    weights = []
    
    for rel in relationships_by_priority:
        weight = get_relationship_weight(rel, depth)
        weights.append((rel, weight))
        print(f"{rel}: {weight:.3f}")
    
    # Check that weights are in descending order (allowing for ties)
    is_ordered = True
    for i in range(len(weights) - 1):
        if weights[i][1] < weights[i + 1][1]:
            print(f"❌ Ordering violation: {weights[i][0]} ({weights[i][1]:.3f}) < {weights[i+1][0]} ({weights[i+1][1]:.3f})")
            is_ordered = False
    
    if is_ordered:
        print("✓ Relationship weights properly ordered")
    else:
        print("❌ Relationship weight ordering incorrect")
    assert is_ordered


def test_extreme_depths():
    """Test behavior at extreme depths"""
    
    extreme_cases = [
        (0, 1.0),    # No decay factor at depth 0
        (5, 0.32768), # 0.8^5 ≈ 0.32768
        (10, 0.10737), # 0.8^10 ≈ 0.10737
    ]
    
    base_relationship = "content_link"  # Base weight 0.9
    base_weight = 0.9
    
    passed = 0
    for depth, expected_factor in extreme_cases:
        weight = get_relationship_weight(base_relationship, depth)
        expected_weight = base_weight * expected_factor
        
        print(f"Depth {depth}: {weight:.5f} (expected: {expected_weight:.5f})")
        
        if abs(weight - expected_weight) < 0.001:
            print("  ✓ Extreme depth handling correct")
            passed += 1
        else:
            print("  ❌ Extreme depth calculation incorrect")
        print()
    
    print(f"Extreme depths: {passed}/{len(extreme_cases)} tests passed")
    assert passed == len(extreme_cases)


def test_composite_scoring():
    """Test how relationship weights might be used in composite scoring"""
    
    # Simulate a scenario where we're scoring multiple relationships
    scenario = [
        ("content_link", 0, 1.0),     # Direct link, high relevance
        ("parent", 1, 0.8),           # Parent section, medium relevance
        ("sibling", 1, 0.6),          # Sibling section, lower relevance
        ("child", 2, 0.7),            # Deep child, medium relevance
    ]
    
    total_score = 0.0
    max_possible = 0.0
    
    print("Composite scoring scenario:")
    for relationship, depth, content_relevance in scenario:
        rel_weight = get_relationship_weight(relationship, depth)
        composite_score = rel_weight * content_relevance
        
        print(f"  {relationship} (depth {depth}): rel_weight={rel_weight:.3f} × content={content_relevance} = {composite_score:.3f}")
        
        total_score += composite_score
        max_possible += content_relevance  # If all relationships had weight 1.0
    
    normalized_score = total_score / max_possible if max_possible > 0 else 0.0
    
    print(f"Total composite score: {total_score:.3f}")
    print(f"Normalized score: {normalized_score:.3f}")
    
    # This is a demonstration test - we expect some reasonable score
    if 0.3 <= normalized_score <= 1.0:
        print("✓ Composite scoring produces reasonable results")
    else:
        print("❌ Composite scoring outside expected range")
    assert 0.3 <= normalized_score <= 1.0


if __name__ == "__main__":
    print("🧪 Testing Relationship Weighting")
    print("=" * 50)
    
    base_test = test_base_relationship_weights()
    decay_test = test_depth_decay()
    ordering_test = test_weight_ordering()
    extreme_test = test_extreme_depths()
    composite_test = test_composite_scoring()
    
    if base_test and decay_test and ordering_test and extreme_test and composite_test:
        print("\n✅ All relationship weighting tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
