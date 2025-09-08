#!/usr/bin/env python3
"""
Unit tests for query intent detection logic.
Tests the pattern matching and intent classification without external dependencies.
"""

import re


def test_query_intent_detection():
    """Test query intent detection logic"""
    
    # Simulate the pattern matching logic from SmartSearchEngine
    tag_patterns = re.compile(r'#(\w+)|tag[s]?\s*[:=]\s*(\w+)|tagged\s+with\s+(\w+)', re.IGNORECASE)
    relationship_patterns = re.compile(r'link[s]?\s+to|connect[s]?\s+to|related\s+to|references?|mentions?|backlink[s]?|graph|network', re.IGNORECASE)
    specific_patterns = re.compile(r'in\s+note\s+|from\s+file\s+|in\s+\w+\s+document|in\s+document|specific[ally]*|exact[ly]*', re.IGNORECASE)
    semantic_patterns = re.compile(r'about|concept|idea|topic|understand|explain|meaning|definition', re.IGNORECASE)
    
    # Test queries based on planet test content
    test_cases = [
        {
            "query": "What is the verification code for Earth?",
            "expected_intent": "semantic",
            "expected_strategy": "vector"
        },
        {
            "query": "Show me notes tagged with planets", 
            "expected_intent": "categorical",
            "expected_strategy": "tag"
        },
        {
            "query": "What links to Mars?",
            "expected_intent": "relationship", 
            "expected_strategy": "graph"
        },
        {
            "query": "Find content in Earth document",
            "expected_intent": "specific",
            "expected_strategy": "vector"
        },
        {
            "query": "Explain the concept of planetary connections",
            "expected_intent": "semantic",
            "expected_strategy": "vector"
        }
    ]
    
    passed = 0
    for test in test_cases:
        query = test["query"]
        
        # Score based on patterns (same logic as SmartSearchEngine)
        intent_scores = {
            "categorical": 0.0,
            "relationship": 0.0, 
            "specific": 0.0,
            "semantic": 0.0
        }
        
        if tag_patterns.search(query):
            intent_scores["categorical"] += 0.8
        if relationship_patterns.search(query):
            intent_scores["relationship"] += 0.7
        if specific_patterns.search(query):
            intent_scores["specific"] += 0.6
        if semantic_patterns.search(query):
            intent_scores["semantic"] += 0.6
            
        # Default semantic search for general queries
        if max(intent_scores.values()) < 0.3:
            intent_scores["semantic"] = 0.8
            
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[primary_intent]
        
        # Map intent to strategy
        strategy_mapping = {
            "categorical": "tag",
            "relationship": "graph",
            "specific": "vector", 
            "semantic": "vector"
        }
        suggested_strategy = strategy_mapping[primary_intent]
        
        print(f"Query: '{query}'")
        print(f"  Detected intent: {primary_intent} (confidence: {confidence:.2f})")
        print(f"  Suggested strategy: {suggested_strategy}")
        print(f"  Expected: {test['expected_intent']} → {test['expected_strategy']}")
        
        if primary_intent == test["expected_intent"]:
            print("  ✓ Intent detection correct")
            passed += 1
        else:
            print("  ❌ Intent detection incorrect")
        print()
    
    print(f"Intent detection: {passed}/{len(test_cases)} tests passed")
    assert passed == len(test_cases)


def test_edge_cases():
    """Test edge cases for intent detection"""
    
    edge_cases = [
        ("", "semantic"),  # Empty query defaults to semantic
        ("a", "semantic"),  # Very short query defaults to semantic 
        ("What are #health goals?", "categorical"),  # Hash tag should trigger categorical
        ("links to health and fitness", "relationship"),  # Links keyword should trigger relationship
        ("find specific content in document", "specific"),  # Multiple specific keywords
    ]
    
    # Same pattern setup as main test
    tag_patterns = re.compile(r'#(\w+)|tag[s]?\s*[:=]\s*(\w+)|tagged\s+with\s+(\w+)', re.IGNORECASE)
    relationship_patterns = re.compile(r'link[s]?\s+to|connect[s]?\s+to|related\s+to|references?|mentions?|backlink[s]?|graph|network', re.IGNORECASE)
    specific_patterns = re.compile(r'in\s+note\s+|from\s+file\s+|in\s+\w+\s+document|in\s+document|specific[ally]*|exact[ly]*', re.IGNORECASE)
    semantic_patterns = re.compile(r'about|concept|idea|topic|understand|explain|meaning|definition', re.IGNORECASE)
    
    passed = 0
    for query, expected_intent in edge_cases:
        intent_scores = {
            "categorical": 0.0,
            "relationship": 0.0, 
            "specific": 0.0,
            "semantic": 0.0
        }
        
        if tag_patterns.search(query):
            intent_scores["categorical"] += 0.8
        if relationship_patterns.search(query):
            intent_scores["relationship"] += 0.7
        if specific_patterns.search(query):
            intent_scores["specific"] += 0.6
        if semantic_patterns.search(query):
            intent_scores["semantic"] += 0.6
            
        # Default semantic search for general queries
        if max(intent_scores.values()) < 0.3:
            intent_scores["semantic"] = 0.8
            
        primary_intent = max(intent_scores, key=intent_scores.get)
        
        if primary_intent == expected_intent:
            passed += 1
            print(f"✓ Edge case '{query}' -> {primary_intent}")
        else:
            print(f"❌ Edge case '{query}' -> {primary_intent}, expected {expected_intent}")
    
    print(f"Edge cases: {passed}/{len(edge_cases)} passed")
    assert passed == len(edge_cases)


if __name__ == "__main__":
    print("🧪 Testing Query Intent Detection")
    print("=" * 50)
    
    main_test = test_query_intent_detection()
    edge_test = test_edge_cases()
    
    if main_test and edge_test:
        print("\n✅ All query intent tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
