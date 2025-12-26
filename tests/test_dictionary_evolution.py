#!/usr/bin/env python3
"""
Test dictionary evolution / learning feature.

Tests the pattern detection and transformation suggestion system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dictionary_learner import DictionaryLearner


def test_technical_term_extraction():
    """Test extracting technical terms from text."""
    print("🧪 Testing technical term extraction...")

    text = """
    This API uses neural network models for machine learning.
    The database stores docker container configs.
    """

    learner = DictionaryLearner()
    terms = learner.extract_technical_terms(text)

    print(f"   Found {len(terms)} unique terms")
    print(f"   Top terms: {dict(terms.most_common(3))}")

    assert 'api' in terms
    assert 'neural' in terms or 'network' in terms
    assert 'database' in terms

    print("✅ Technical term extraction test PASSED!\n")
    return True


def test_bigram_extraction():
    """Test bigram extraction."""
    print("🧪 Testing bigram extraction...")

    text = "neural network architecture machine learning pipeline machine learning model"

    learner = DictionaryLearner()
    bigrams = learner.extract_bigrams(text)

    print(f"   Found {len(bigrams)} unique bigrams")
    print(f"   Top bigrams: {dict(bigrams.most_common(3))}")

    assert 'machine learning' in bigrams
    assert bigrams['machine learning'] == 2  # Appears twice

    print("✅ Bigram extraction test PASSED!\n")
    return True


def test_character_name_suggestions():
    """Test character name generation from technical terms."""
    print("🧪 Testing character name suggestions...")

    # Create a learner and mock term frequency
    learner = DictionaryLearner()

    from collections import Counter
    terms = Counter({
        'docker': 5,
        'api': 7,
        'kubernetes': 3,
        'rare_term': 1  # Should be ignored (too rare)
    })

    suggestions = learner.suggest_character_names(terms)

    print(f"   Generated {len(suggestions)} suggestions:")
    for term, name, reason in suggestions[:3]:
        print(f"     • {term} → {name}")

    # Check that common terms got suggestions
    suggested_terms = [s[0] for s in suggestions]
    assert 'docker' in suggested_terms or 'api' in suggested_terms

    # Rare term should not be suggested
    assert 'rare_term' not in suggested_terms

    print("✅ Character name suggestion test PASSED!\n")
    return True


def test_action_transformations():
    """Test action verb transformation suggestions."""
    print("🧪 Testing action transformation suggestions...")

    text = """
    First we initialize the system. Then we validate the input.
    The compiler compiles the code. We validate the output again.
    """

    learner = DictionaryLearner()
    suggestions = learner.suggest_action_transformations(text)

    print(f"   Generated {len(suggestions)} suggestions:")
    for verb, transformation, reason in suggestions:
        print(f"     • {verb} → {transformation}")

    # Check that repeated verbs got suggested
    suggested_verbs = [s[0] for s in suggestions]
    assert 'validate' in suggested_verbs  # Appears twice

    print("✅ Action transformation test PASSED!\n")
    return True


def test_concept_transformations():
    """Test concept phrase transformation suggestions."""
    print("🧪 Testing concept transformation suggestions...")

    from collections import Counter
    bigrams = Counter({
        'machine learning': 3,
        'neural network': 2,
        'rare phrase': 1
    })

    learner = DictionaryLearner()
    suggestions = learner.suggest_concept_transformations(bigrams)

    print(f"   Generated {len(suggestions)} suggestions:")
    for phrase, transformation, reason in suggestions:
        print(f"     • {phrase} → {transformation}")

    # Check that common phrases got suggested
    suggested_phrases = [s[0] for s in suggestions]
    assert 'machine learning' in suggested_phrases

    print("✅ Concept transformation test PASSED!\n")
    return True


def test_full_analysis():
    """Test full analysis pipeline."""
    print("🧪 Testing full analysis pipeline...")

    github_text = """
    This neural network implementation uses Docker containers.
    The API handles machine learning requests. We initialize
    the database and validate all inputs. The neural network
    processes data and the machine learning pipeline optimizes
    results. Docker deployment is automated via CI/CD.
    """

    learner = DictionaryLearner()
    suggestions = learner.analyze_and_suggest(github_text)

    print(f"   Categories analyzed: {list(suggestions.keys())}")

    total = sum(len(sug_list) for sug_list in suggestions.values())
    print(f"   Total suggestions: {total}")

    # Should have at least some suggestions
    assert total > 0, "Should generate at least one suggestion"

    # Show sample suggestions
    for category, sug_list in suggestions.items():
        if sug_list:
            print(f"   {category}: {len(sug_list)} suggestions")
            print(f"     Example: {sug_list[0][0]} → {sug_list[0][1]}")

    print("✅ Full analysis test PASSED!\n")
    return True


def test_dictionary_persistence():
    """Test saving and loading learned dictionary."""
    print("🧪 Testing dictionary persistence...")

    import tempfile
    import os

    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()

    try:
        # Create learner and add transformation
        learner1 = DictionaryLearner(dictionary_file=temp_file.name)
        learner1.add_transformation("testword", "testtransform", "test")

        # Load in new learner
        learner2 = DictionaryLearner(dictionary_file=temp_file.name)

        assert 'testword' in learner2.learned_dict
        assert learner2.learned_dict['testword'] == 'testtransform'

        print("   ✅ Dictionary saved and loaded successfully")
        print(f"   Learned dict: {learner2.learned_dict}")

        print("✅ Dictionary persistence test PASSED!\n")
        return True

    finally:
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def main():
    """Run all dictionary evolution tests."""
    print("\n🌱 DICTIONARY EVOLUTION TESTS\n")
    print("="*60)

    tests = [
        ("Technical term extraction", test_technical_term_extraction),
        ("Bigram extraction", test_bigram_extraction),
        ("Character name suggestions", test_character_name_suggestions),
        ("Action transformations", test_action_transformations),
        ("Concept transformations", test_concept_transformations),
        ("Full analysis pipeline", test_full_analysis),
        ("Dictionary persistence", test_dictionary_persistence),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}\n")
            import traceback
            traceback.print_exc()

    print("="*60)
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("✅ ALL TESTS PASSED! 🌱\n")
    else:
        print(f"❌ {failed} tests failed\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
