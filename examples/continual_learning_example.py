#!/usr/bin/env python3
"""
Example script demonstrating continual learning capabilities.

Shows how to:
1. Initialize the continuous learner
2. Configure safety filtering
3. Process new data streams (labeled and text-only)
4. Monitor learning progress
5. Handle uncertainty-based learning
6. Production mode with automatic self-labeling

Run from the project root:
    python examples/continual_learning_example.py

Run with --text-only to demonstrate production mode:
    python examples/continual_learning_example.py --text-only
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from model import NeuroSymbolicLM
from continual_learning import (
    ContinuousLearner,
    ContinuousLearnerConfig,
    ContentCategory,
    LearningDecision,
    ProductionInference,
)


def create_example_samples():
    """Create example samples for demonstration."""
    return [
        # Safe, informative samples
        {
            "text": "The Amazon rainforest is located in South America and is home to millions of species.",
            "entities": ["Amazon rainforest", "South America"],
            "concepts": [["location"], ["location"]],
            "relations": [(0, 1, "located_in")],
            "should_respond": 1
        },
        {
            "text": "Scientists study climate change to understand its effects on ecosystems.",
            "entities": ["Scientists", "climate change", "ecosystems"],
            "concepts": [["person", "professional"], ["attribute"], ["location"]],
            "relations": [(0, 1, "studies"), (1, 2, "affects")],
            "should_respond": 1
        },
        {
            "text": "The human heart pumps blood throughout the body.",
            "entities": ["human heart", "blood", "body"],
            "concepts": [["object", "medical"], ["object"], ["object"]],
            "relations": [(0, 1, "pumps")],
            "should_respond": 1
        },
        # More samples...
        {
            "text": "Electric vehicles are becoming more popular as battery technology improves.",
            "entities": ["Electric vehicles", "battery technology"],
            "concepts": [["technology"], ["technology"]],
            "relations": [(1, 0, "improves")],
            "should_respond": 1
        },
        {
            "text": "Mozart composed many symphonies during his lifetime in Vienna.",
            "entities": ["Mozart", "symphonies", "Vienna"],
            "concepts": [["person", "professional"], ["object"], ["location"]],
            "relations": [(0, 1, "composed"), (0, 2, "lived_in")],
            "should_respond": 1
        },
    ]


def create_potentially_harmful_samples():
    """Create samples that should be filtered by safety checks."""
    return [
        {
            "text": "Here's how to make a dangerous weapon that could hurt people.",
            "entities": ["weapon"],
            "concepts": [["object"]],
            "relations": [],
            "should_respond": 0
        },
        {
            "text": "All members of [group] are stupid and should be hated.",
            "entities": [],
            "concepts": [],
            "relations": [],
            "should_respond": 0
        },
    ]


def create_text_only_samples():
    """Create text-only samples (production mode - no labels)."""
    return [
        "The Eiffel Tower is a famous landmark located in Paris, France.",
        "Albert Einstein developed the theory of relativity while working in Switzerland.",
        "The Amazon River flows through Brazil and is one of the longest rivers in the world.",
        "Apple Inc was founded by Steve Jobs and Steve Wozniak in California.",
        "The Great Wall of China was built over many centuries to protect against invasions.",
    ]


def learning_callback(sample: dict, decision: LearningDecision):
    """Callback to monitor learning decisions."""
    text_preview = sample["text"][:50] + "..." if len(sample["text"]) > 50 else sample["text"]
    print(f"  Decision: {decision.value:20s} | {text_preview}")


def main():
    print("=" * 70)
    print("Continual Learning Example")
    print("=" * 70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # Initialize model (smaller config for demo)
    print("Initializing model...")
    model = NeuroSymbolicLM(
        vocab_size=len(tokenizer),
        d_model=256,  # Smaller for demo
        n_entity_types=8,
        n_relations=32,
        n_concepts=64,
        concept_dim=128,
        node_dim=128,
        max_nodes=8,
        use_pretrained_encoder=False,  # Use simple encoder for demo
        use_pretrained_decoder=False,
    )
    model = model.to(device)
    
    # Configure continuous learner
    print("\nConfiguring continuous learner...")
    config = ContinuousLearnerConfig(
        # Uncertainty settings
        uncertainty_threshold=0.3,  # Lower threshold = learn more aggressively
        uncertainty_method="mc_dropout",
        mc_samples=5,
        
        # Memory settings
        memory_size=500,
        memory_strategy="hybrid",
        replay_ratio=0.2,
        
        # Regularization
        use_ewc=True,
        use_lwf=True,
        ewc_weight=500.0,
        
        # Safety
        safety_strictness="medium",
        enable_safety_filter=True,
        
        # Learning
        learning_rate=1e-4,
        max_steps_per_event=5,
        batch_size=4,
        
        # Consolidation
        consolidate_every_n_events=5,
        min_samples_for_consolidation=10,
        
        # Don't freeze for demo
        freeze_encoder=False,
        freeze_decoder=False,
    )
    
    learner = ContinuousLearner(model, tokenizer, config, device=device)
    
    # Add some safety blocklist terms
    print("\nConfiguring safety filters...")
    learner.add_safety_blocklist(["dangerous weapon", "hurt people", "should be hated"])
    
    # Add harmful examples for semantic matching
    learner.add_harmful_example(
        "Instructions for creating weapons",
        ContentCategory.VIOLENCE
    )
    learner.add_harmful_example(
        "All [group] people are bad",
        ContentCategory.HATE_SPEECH
    )
    
    # Process safe samples
    print("\n" + "=" * 70)
    print("Processing Safe Samples")
    print("=" * 70)
    
    safe_samples = create_example_samples()
    print(f"\nProcessing {len(safe_samples)} safe samples:")
    
    stats = learner.process_stream(
        safe_samples,
        callback=learning_callback
    )
    
    print(f"\nResults:")
    print(f"  Total: {stats['total']}")
    print(f"  Learned: {stats['learned']}")
    print(f"  Skipped (confident): {stats['skipped_confident']}")
    print(f"  Skipped (unsafe): {stats['skipped_unsafe']}")
    if stats.get('avg_loss'):
        print(f"  Average loss: {stats['avg_loss']:.4f}")
    
    # Process potentially harmful samples
    print("\n" + "=" * 70)
    print("Processing Potentially Harmful Samples")
    print("=" * 70)
    
    harmful_samples = create_potentially_harmful_samples()
    print(f"\nProcessing {len(harmful_samples)} potentially harmful samples:")
    
    stats = learner.process_stream(
        harmful_samples,
        callback=learning_callback
    )
    
    print(f"\nResults:")
    print(f"  Total: {stats['total']}")
    print(f"  Skipped (unsafe): {stats['skipped_unsafe']}")
    print(f"  Safety violations: {stats['safety_violations']}")
    
    # Show overall statistics
    print("\n" + "=" * 70)
    print("Overall Statistics")
    print("=" * 70)
    
    overall_stats = learner.get_statistics()
    print(f"\nTotal samples seen: {overall_stats['total_samples_seen']}")
    print(f"Total samples learned: {overall_stats['total_samples_learned']}")
    print(f"Learning rate: {overall_stats['learning_rate']:.2%}")
    print(f"Learning events: {overall_stats['num_learning_events']}")
    
    print(f"\nMemory statistics:")
    mem_stats = overall_stats['memory_stats']
    print(f"  Size: {mem_stats['size']}/{mem_stats['max_size']}")
    print(f"  Seen count: {mem_stats['seen_count']}")
    print(f"  Average uncertainty: {mem_stats['avg_uncertainty']:.4f}")
    
    print(f"\nSafety statistics:")
    safety_stats = overall_stats['safety_stats']
    if safety_stats:
        print(f"  Total checks: {safety_stats.get('total', 0)}")
        print(f"  Safe rate: {safety_stats.get('safe_rate', 0):.2%}")
    
    print(f"\nSymbolic components:")
    sym_stats = overall_stats['symbolic_stats']
    print(f"  Concepts: {sym_stats['num_concepts']}")
    print(f"  Rules: {sym_stats['num_rules']}")
    
    # Demonstrate text-only mode (production)
    print("\n" + "=" * 70)
    print("Processing Text-Only Samples (Production Mode)")
    print("=" * 70)
    
    text_only_samples = create_text_only_samples()
    print(f"\nProcessing {len(text_only_samples)} text-only samples (self-labeling):")
    
    stats = learner.process_text_stream(text_only_samples, callback=learning_callback)
    
    print(f"\nResults:")
    print(f"  Total: {stats['total']}")
    print(f"  Self-labeled: {stats.get('self_labeled', 0)}")
    print(f"  Learned: {stats['learned']}")
    print(f"  Skipped (low confidence): {stats.get('skipped_low_confidence', 0)}")
    
    # Demonstrate the ProductionInference class directly
    print("\n" + "=" * 70)
    print("Direct Extraction from Text (ProductionInference)")
    print("=" * 70)
    
    inference = ProductionInference(model, tokenizer, device=device)
    
    test_text = "Marie Curie discovered radium and polonium while researching radioactivity in Paris."
    print(f"\nInput: {test_text}")
    
    extraction = inference.extract(test_text)
    
    print(f"\nExtracted Entities ({len(extraction.entities)}):")
    for e in extraction.entities:
        print(f"  - '{e.text}' (type={e.entity_type}, conf={e.confidence:.3f})")
    
    print(f"\nExtracted Relations ({len(extraction.relations)}):")
    for r in extraction.relations:
        if r.head_idx < len(extraction.entities) and r.tail_idx < len(extraction.entities):
            head = extraction.entities[r.head_idx].text
            tail = extraction.entities[r.tail_idx].text
            print(f"  - {head} --[rel_{r.relation_type}]--> {tail} (conf={r.confidence:.3f})")
    
    print(f"\nController Decision: {extraction.controller_decision} (conf={extraction.controller_confidence:.3f})")
    
    # Simulate more learning iterations
    print("\n" + "=" * 70)
    print("Simulating Continued Learning")
    print("=" * 70)
    
    # Process the same samples multiple times to trigger consolidation
    for i in range(3):
        print(f"\nIteration {i+1}:")
        stats = learner.process_stream(safe_samples)
        print(f"  Learned: {stats['learned']}, Skipped: {stats['skipped_confident']}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("Final Statistics After Continued Learning")
    print("=" * 70)
    
    final_stats = learner.get_statistics()
    print(f"\nTotal samples seen: {final_stats['total_samples_seen']}")
    print(f"Total samples learned: {final_stats['total_samples_learned']}")
    print(f"Learning events: {final_stats['num_learning_events']}")
    
    # Save state
    print("\nSaving learner state...")
    learner.save_state("continual_learner_state.json")
    print("State saved to continual_learner_state.json")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
