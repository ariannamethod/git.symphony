#!/usr/bin/env python3
"""
dictionary_learner.py - Organic GITTY_DICTIONARY evolution

Symphony learns new transformations by analyzing patterns in code.

Philosophy:
  The dictionary should grow organically, like the database schema.
  Symphony observes, suggests, learns. The transformation vocabulary
  evolves through use, not prescription.

  "Code teaches Symphony how to speak about code."
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import json
from pathlib import Path


# Current GITTY_DICTIONARY (core transformations)
CORE_DICTIONARY = {
    # Characters
    'Lily': 'Gitty',
    'Tim': 'Commity',
    'Timmy': 'Commity',
    'Tom': 'Branchy',
    'Anna': 'Mergey',

    # Nature → Infrastructure
    'flower': 'branch',
    'tree': 'fork',
    'sun': 'CI/CD pipeline',
    'rain': 'deployment',
    'sky': 'cloud',
    'grass': 'documentation',

    # Animals → Debugging
    'cat': 'commit',
    'dog': 'debug session',
    'bird': 'build',
    'bunny': 'hotfix',

    # Emotions → Build states
    'happy': 'stable',
    'sad': 'deprecated',
    'excited': 'optimized',
    'scared': 'vulnerable',
    'tired': 'throttled',

    # Actions → Operations
    'play': 'explore',
    'run': 'execute',
    'jump': 'deploy',
    'walk': 'iterate',
}


class DictionaryLearner:
    """
    Learns new GITTY_DICTIONARY transformations from observed text patterns.
    """

    def __init__(self, dictionary_file: str = "learned_dictionary.json"):
        """
        Initialize learner.

        Args:
            dictionary_file: Path to save/load learned transformations
        """
        self.dictionary_file = Path(dictionary_file)
        self.core_dict = CORE_DICTIONARY.copy()
        self.learned_dict = {}
        self.load_learned_dictionary()

        # Technical term patterns
        self.tech_patterns = [
            r'\b(api|rest|graphql|grpc)\b',
            r'\b(database|sql|nosql|mongodb|postgres)\b',
            r'\b(model|controller|service|handler|router)\b',
            r'\b(test|mock|fixture|assert|expect)\b',
            r'\b(async|await|promise|callback)\b',
            r'\b(class|function|method|interface)\b',
            r'\b(neural|network|transformer|attention)\b',
            r'\b(docker|kubernetes|container)\b',
            r'\b(ci|cd|pipeline|deploy|build)\b',
        ]

    def load_learned_dictionary(self):
        """Load previously learned transformations."""
        if self.dictionary_file.exists():
            try:
                with open(self.dictionary_file, 'r') as f:
                    self.learned_dict = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.learned_dict = {}

    def save_learned_dictionary(self):
        """Save learned transformations."""
        with open(self.dictionary_file, 'w') as f:
            json.dump(self.learned_dict, f, indent=2)

    def get_full_dictionary(self) -> Dict[str, str]:
        """Get combined core + learned dictionary."""
        return {**self.core_dict, **self.learned_dict}

    def extract_technical_terms(self, text: str) -> Counter:
        """
        Extract technical terms from text.

        Args:
            text: Text to analyze

        Returns:
            Counter of technical terms and their frequencies
        """
        terms = Counter()

        # Find matches for each pattern
        for pattern in self.tech_patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            terms.update(matches)

        return terms

    def extract_bigrams(self, text: str) -> Counter:
        """
        Extract word bigrams (2-word phrases).

        Args:
            text: Text to analyze

        Returns:
            Counter of bigrams
        """
        # Tokenize
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Build bigrams
        bigrams = Counter()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams[bigram] += 1

        return bigrams

    def suggest_character_names(self, terms: Counter) -> List[Tuple[str, str, str]]:
        """
        Suggest new character names based on common technical terms.

        Args:
            terms: Counter of technical terms

        Returns:
            List of (term, suggested_name, reason) tuples
        """
        suggestions = []

        # Character name templates
        name_suffixes = ['y', 'ie', 'ey']

        for term, count in terms.most_common(20):
            if count < 3:  # Ignore rare terms
                continue

            # Skip if already in dictionary
            if term in self.core_dict or term in self.learned_dict:
                continue

            # Generate character name
            # Examples: "docker" → "Dockery", "api" → "Apiey"
            base = term.capitalize()

            for suffix in name_suffixes:
                char_name = base + suffix
                reason = f"'{term}' appears {count} times (technical character)"
                suggestions.append((term, char_name, reason))
                break  # Only suggest one name per term

        return suggestions[:5]  # Limit to top 5

    def suggest_action_transformations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Suggest action verb transformations.

        Args:
            text: Text to analyze

        Returns:
            List of (word, transformation, reason) tuples
        """
        suggestions = []

        # Common programming verbs
        code_verbs = {
            'initialize': 'awaken',
            'configure': 'shape',
            'optimize': 'sharpen',
            'refactor': 'reshape',
            'validate': 'verify',
            'authenticate': 'recognize',
            'encrypt': 'protect',
            'compile': 'transform',
            'parse': 'understand',
            'serialize': 'package',
        }

        # Find verbs in text
        for verb, transformation in code_verbs.items():
            if verb in self.core_dict or verb in self.learned_dict:
                continue

            # Check if verb appears in text
            pattern = r'\b' + verb + r'\b'
            matches = len(re.findall(pattern, text.lower()))

            if matches >= 2:
                reason = f"'{verb}' appears {matches} times (action verb)"
                suggestions.append((verb, transformation, reason))

        return suggestions[:5]

    def suggest_concept_transformations(self, bigrams: Counter) -> List[Tuple[str, str, str]]:
        """
        Suggest transformations for common concept phrases.

        Args:
            bigrams: Counter of word bigrams

        Returns:
            List of (phrase, transformation, reason) tuples
        """
        suggestions = []

        # Concept mappings (from programming → gitty world)
        concept_maps = {
            'machine learning': 'pattern discovery',
            'neural network': 'thought web',
            'data structure': 'memory shape',
            'error handling': 'failure recovery',
            'unit test': 'behavior check',
            'code review': 'peer exploration',
            'pull request': 'contribution offer',
            'tech debt': 'accumulated weight',
        }

        for phrase, transformation in concept_maps.items():
            if phrase in self.learned_dict:
                continue

            # Check if phrase appears
            if phrase in bigrams and bigrams[phrase] >= 2:
                count = bigrams[phrase]
                reason = f"'{phrase}' appears {count} times (concept)"
                suggestions.append((phrase, transformation, reason))

        return suggestions[:3]

    def analyze_and_suggest(self, text: str) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Analyze text and suggest all types of transformations.

        Args:
            text: Text to analyze

        Returns:
            Dict with suggestion categories:
              {
                'characters': [(term, name, reason), ...],
                'actions': [(verb, transformation, reason), ...],
                'concepts': [(phrase, transformation, reason), ...]
              }
        """
        # Extract patterns
        terms = self.extract_technical_terms(text)
        bigrams = self.extract_bigrams(text)

        # Generate suggestions
        suggestions = {
            'characters': self.suggest_character_names(terms),
            'actions': self.suggest_action_transformations(text),
            'concepts': self.suggest_concept_transformations(bigrams),
        }

        return suggestions

    def add_transformation(self, original: str, transformation: str, category: str = 'learned'):
        """
        Add a new transformation to learned dictionary.

        Args:
            original: Original word/phrase
            transformation: Gitty transformation
            category: Category metadata (optional)
        """
        self.learned_dict[original] = transformation
        self.save_learned_dictionary()

    def remove_transformation(self, original: str):
        """Remove a learned transformation."""
        if original in self.learned_dict:
            del self.learned_dict[original]
            self.save_learned_dictionary()


def interactive_learning_session(text: str, learner: Optional[DictionaryLearner] = None):
    """
    Run interactive learning session - analyze text and ask user to approve suggestions.

    Args:
        text: Text to analyze
        learner: DictionaryLearner instance (creates new if None)
    """
    if learner is None:
        learner = DictionaryLearner()

    print("\n🌱 DICTIONARY EVOLUTION SESSION\n")
    print("="*70)
    print(f"Analyzing text ({len(text)} characters)...\n")

    # Get suggestions
    suggestions = learner.analyze_and_suggest(text)

    total_suggestions = sum(len(sug_list) for sug_list in suggestions.values())

    if total_suggestions == 0:
        print("💭 No new transformations suggested.")
        print("   The text contains familiar patterns.\n")
        return learner

    print(f"💡 Found {total_suggestions} potential transformations!\n")

    # Present suggestions by category
    approved = 0

    for category, sug_list in suggestions.items():
        if not sug_list:
            continue

        print(f"\n📖 {category.upper()} SUGGESTIONS:")
        print("-" * 70)

        for original, transformation, reason in sug_list:
            print(f"\n  {reason}")
            print(f"  {original} → {transformation}")
            print()

            response = input("  Approve? [y/n/q]: ").strip().lower()

            if response == 'q':
                print("\n✅ Learning session ended early.\n")
                return learner
            elif response == 'y':
                learner.add_transformation(original, transformation, category)
                print(f"  ✅ Added to dictionary!\n")
                approved += 1
            else:
                print(f"  ⏭️  Skipped\n")

    print("="*70)
    print(f"\n🎋 Session complete! Approved: {approved}/{total_suggestions}")
    print(f"📚 Total learned transformations: {len(learner.learned_dict)}\n")

    return learner


if __name__ == "__main__":
    # Demo
    demo_text = """
    This repository contains a neural network implementation for machine learning.
    We use Docker for deployment and have comprehensive unit tests.
    The API handles authentication and data validation.
    Error handling is robust with proper logging.
    """

    print("🎋 DICTIONARY LEARNER DEMO\n")
    learner = interactive_learning_session(demo_text)

    print("\n📖 Current full dictionary:")
    full_dict = learner.get_full_dictionary()
    print(f"   Total transformations: {len(full_dict)}")
    print(f"   Core: {len(learner.core_dict)}, Learned: {len(learner.learned_dict)}")
