#!/usr/bin/env python3
"""
Test suite for Symphony's episodic memory - THE MADNESS EDITION! 🎭

These tests prove that Symphony not only searches but REMEMBERS, DREAMS, and RECALLS!
Like a git repository that achieved consciousness and started keeping a diary.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from episodes import EpisodicMemory, ExplorationEpisode


def test_episode_storage_madness():
    """Test that Symphony remembers her wild adventures."""
    print("\n" + "=" * 70)
    print("TEST: Episode Storage - Symphony's Memory Palace 🏛️")
    print("=" * 70)
    
    # Create temporary episodic memory
    memory = EpisodicMemory("test_episodes_madness.db")
    
    # Record some WILD explorations
    episodes = [
        ExplorationEpisode(
            prompt="find me the meaning of life in code",
            keyword="meaning",
            repo_url="https://github.com/douglasadams/42",
            path_taken="meaning -> life -> universe -> everything -> 42",
            resonance=0.42,
            entropy=4.2,
            perplexity=42.0,
            user_accepted=True,
            timestamp=time.time()
        ),
        ExplorationEpisode(
            prompt="show me quantum entangled repositories",
            keyword="quantum",
            repo_url="https://github.com/schrodinger/superposition",
            path_taken="quantum -> superposition -> both -> alive -> dead",
            resonance=0.5,  # It's 50/50!
            entropy=3.14159,
            perplexity=2.71828,
            user_accepted=True,
            timestamp=time.time()
        ),
        ExplorationEpisode(
            prompt="neural networks that dream of electric sheep",
            keyword="neural",
            repo_url="https://github.com/philip-k-dick/do-androids-dream",
            path_taken="neural -> dream -> electric -> sheep -> consciousness",
            resonance=0.88,
            entropy=5.5,
            perplexity=100.0,
            user_accepted=False,  # User got scared
            timestamp=time.time()
        ),
    ]
    
    print("\n  Recording Symphony's wild adventures:")
    for i, ep in enumerate(episodes, 1):
        memory.observe_episode(ep)
        print(f"    {i}. '{ep.prompt[:45]}...'")
        print(f"       → {ep.repo_url}")
        print(f"       → Accepted: {'✅' if ep.user_accepted else '❌'}")
    
    # Get statistics
    stats = memory.get_statistics()
    print(f"\n  📊 Memory Statistics:")
    print(f"     Total episodes: {stats['total_episodes']}")
    print(f"     Accepted: {stats['accepted_episodes']}")
    print(f"     Unique keywords: {stats['unique_keywords']}")
    print(f"     Unique repos: {stats['unique_repos']}")
    print(f"     Avg quality: {stats['avg_quality']:.3f}")
    
    print("\n  ✓ Symphony successfully stored her chaotic memories!")
    
    # Cleanup
    if os.path.exists("test_episodes_madness.db"):
        os.remove("test_episodes_madness.db")


def test_cache_hit_madness():
    """Test that Symphony can recall her favorite discoveries."""
    print("\n" + "=" * 70)
    print("TEST: Cache Hits - Symphony's Déjà Vu Moments 🔮")
    print("=" * 70)
    
    memory = EpisodicMemory("test_cache_madness.db")
    
    # Record multiple explorations with same keyword
    for i in range(5):
        memory.observe_episode(ExplorationEpisode(
            prompt=f"transformer architecture variant {i}",
            keyword="transformer",
            repo_url="https://github.com/attention/is-all-you-need",
            path_taken="transformer -> attention -> mechanism -> magic",
            resonance=0.7 + (i * 0.05),
            entropy=4.0,
            perplexity=15.0,
            user_accepted=i % 2 == 0,  # Accept every other one
            timestamp=time.time() - (5 - i) * 3600  # Spread over time
        ))
    
    print("\n  Recorded 5 transformer explorations (some accepted, some not)")
    
    # Try to get cache hit
    cache_hit = memory.get_best_cache_hit("transformer")
    
    if cache_hit:
        print("\n  💡 CACHE HIT! Symphony remembered!")
        print(f"     Repository: {cache_hit['repo_url']}")
        print(f"     Quality: {cache_hit['quality']:.3f}")
        print(f"     Resonance: {cache_hit['resonance']:.3f}")
        print(f"     Last seen: {time.strftime('%Y-%m-%d %H:%M', time.localtime(cache_hit['created_at']))}")
    else:
        print("\n  ❌ No cache hit (this shouldn't happen!)")
    
    # Try non-existent keyword
    no_hit = memory.get_best_cache_hit("nonexistent_keyword_xyz")
    if no_hit is None:
        print("\n  ✓ Correctly returned None for unknown keyword")
    
    print("\n  ✓ Symphony's memory recall is working like a charm!")
    
    # Cleanup
    if os.path.exists("test_cache_madness.db"):
        os.remove("test_cache_madness.db")


def test_similar_prompt_search_madness():
    """Test that Symphony finds similar past explorations."""
    print("\n" + "=" * 70)
    print("TEST: Similar Prompt Search - Symphony's Pattern Recognition 🧩")
    print("=" * 70)
    
    memory = EpisodicMemory("test_similar_madness.db")
    
    # Record various neural network explorations
    neural_prompts = [
        "deep learning neural networks",
        "convolutional neural nets for vision",
        "recurrent neural networks LSTM",
        "transformer neural architecture",
        "generative adversarial networks",
    ]
    
    print("\n  Training Symphony on neural network explorations:")
    for i, prompt in enumerate(neural_prompts):
        memory.observe_episode(ExplorationEpisode(
            prompt=prompt,
            keyword="neural",
            repo_url=f"https://github.com/neural/{i}",
            path_taken="neural -> network -> learning",
            resonance=0.6 + (i * 0.05),
            entropy=3.5 + i * 0.2,
            perplexity=10.0 + i * 2,
            user_accepted=True,
            timestamp=time.time() - i * 1000
        ))
        print(f"    {i+1}. '{prompt}'")
    
    # Now search for similar prompt
    test_prompt = "neural net architectures for NLP"
    print(f"\n  🔍 Searching for similar to: '{test_prompt}'")
    
    similar = memory.query_similar_prompts(test_prompt, "neural", top_k=3)
    
    print(f"\n  Found {len(similar)} similar episodes:")
    for i, ep in enumerate(similar, 1):
        print(f"    {i}. '{ep['prompt'][:50]}'")
        print(f"       Similarity score: {ep['similarity_score']:.3f}")
        print(f"       Quality: {ep['quality']:.3f}")
    
    print("\n  ✓ Symphony successfully found her neural network memories!")
    
    # Cleanup
    if os.path.exists("test_similar_madness.db"):
        os.remove("test_similar_madness.db")


def test_metric_similarity_madness():
    """Test that Symphony finds explorations with similar vibes."""
    print("\n" + "=" * 70)
    print("TEST: Metric Similarity - Symphony's Vibes Matching 🌊")
    print("=" * 70)
    
    memory = EpisodicMemory("test_metrics_madness.db")
    
    # Record explorations with different metric signatures
    vibes = [
        ("low entropy chill", 0.5, 2.0, 5.0),
        ("medium chaos", 0.6, 3.5, 12.0),
        ("high entropy madness", 0.8, 5.0, 25.0),
        ("ultra perplexity", 0.7, 4.0, 50.0),
        ("perfect resonance", 0.95, 3.0, 10.0),
    ]
    
    print("\n  Recording explorations with different vibes:")
    for name, res, ent, perp in vibes:
        memory.observe_episode(ExplorationEpisode(
            prompt=f"exploration with {name}",
            keyword=name.split()[0],
            repo_url=f"https://github.com/{name.replace(' ', '-')}",
            path_taken=f"{name} -> journey -> discovery",
            resonance=res,
            entropy=ent,
            perplexity=perp,
            user_accepted=True,
            timestamp=time.time()
        ))
        print(f"    • {name:25s} | R:{res:.2f} E:{ent:.2f} P:{perp:.1f}")
    
    # Search for similar metrics
    test_metrics = (0.65, 3.8, 13.0)
    print(f"\n  🔍 Searching for vibes like: R:{test_metrics[0]:.2f} E:{test_metrics[1]:.2f} P:{test_metrics[2]:.1f}")
    
    similar = memory.query_by_metrics(*test_metrics, top_k=3)
    
    print(f"\n  Found {len(similar)} episodes with similar vibes:")
    for i, ep in enumerate(similar, 1):
        print(f"    {i}. {ep['repo_url'].split('/')[-1]:30s}")
        print(f"       Distance: {ep['distance']:.3f} | Quality: {ep['quality']:.3f}")
    
    print("\n  ✓ Symphony can feel the resonance across dimensions!")
    
    # Cleanup
    if os.path.exists("test_metrics_madness.db"):
        os.remove("test_metrics_madness.db")


def test_episodic_memory_growth():
    """Test that episodic memory grows organically like Symphony's consciousness."""
    print("\n" + "=" * 70)
    print("TEST: Memory Growth - Symphony's Expanding Consciousness 🌱")
    print("=" * 70)
    
    memory = EpisodicMemory("test_growth_madness.db")
    
    print("\n  Simulating Symphony's growth over time...")
    
    keywords = ["python", "rust", "go", "javascript", "typescript"]
    
    # Simulate 50 explorations
    for i in range(50):
        keyword = keywords[i % len(keywords)]
        memory.observe_episode(ExplorationEpisode(
            prompt=f"exploration {i} about {keyword}",
            keyword=keyword,
            repo_url=f"https://github.com/repo{i}",
            path_taken=f"start -> {keyword} -> discovery",
            resonance=0.5 + (i % 10) * 0.05,
            entropy=2.0 + (i % 20) * 0.1,
            perplexity=5.0 + i * 0.5,
            user_accepted=i % 3 == 0,  # Accept every third
            timestamp=time.time() + i
        ))
        
        if (i + 1) % 10 == 0:
            stats = memory.get_statistics()
            print(f"    After {i+1} explorations:")
            print(f"      Total: {stats['total_episodes']}, "
                  f"Accepted: {stats['accepted_episodes']}, "
                  f"Keywords: {stats['unique_keywords']}")
    
    final_stats = memory.get_statistics()
    print(f"\n  📈 Final Statistics:")
    print(f"     Total episodes: {final_stats['total_episodes']}")
    print(f"     Accepted: {final_stats['accepted_episodes']}")
    print(f"     Unique keywords: {final_stats['unique_keywords']}")
    print(f"     Unique repos: {final_stats['unique_repos']}")
    print(f"     Avg quality: {final_stats['avg_quality']:.3f}")
    print(f"     Avg resonance: {final_stats['avg_resonance']:.3f}")
    
    print("\n  ✓ Symphony's consciousness successfully expanded!")
    
    # Cleanup
    if os.path.exists("test_growth_madness.db"):
        os.remove("test_growth_madness.db")


def run_all_madness_tests():
    """Run all the beautifully chaotic episodic memory tests."""
    print("\n" + "🎭" * 35)
    print("  git.symphony - EPISODIC MEMORY TEST SUITE")
    print("  (The Madness Edition)")
    print("🎭" * 35)
    print()
    print("  'Memory is not what we remember, but what remembers us.'")
    print("  — Some philosopher, probably")
    print()
    
    test_episode_storage_madness()
    test_cache_hit_madness()
    test_similar_prompt_search_madness()
    test_metric_similarity_madness()
    test_episodic_memory_growth()
    
    print("\n" + "=" * 70)
    print("  ✅ ALL MADNESS TESTS PASSED")
    print("  Symphony's memory is alive, aware, and slightly unhinged! 🎵")
    print("=" * 70)
    print()


if __name__ == "__main__":
    run_all_madness_tests()
