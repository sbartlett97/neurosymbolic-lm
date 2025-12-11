"""Evaluation utilities for generation quality assessment.

Includes:
- BLEU score computation
- Entity F1 score
- Generation evaluation
"""

from typing import List, Dict, Optional, Tuple
from collections import Counter
import math
import torch


def compute_bleu_score(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    Compute BLEU score between reference and hypothesis.
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        max_n: Maximum n-gram order
        weights: Weights for each n-gram order (default: uniform)
    
    Returns:
        BLEU score between 0 and 1
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    
    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_get_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_get_ngrams(hyp_tokens, n))
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        clipped_count = sum(
            min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in hyp_ngrams.items()
        )
        total_count = sum(hyp_ngrams.values())
        
        precision = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(precision)
    
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
    bleu = bp * math.exp(log_precision)
    
    return bleu


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_entity_f1(
    reference_entities: List[str],
    predicted_entities: List[str]
) -> Dict[str, float]:
    """
    Compute entity-level precision, recall, and F1 score.
    
    Args:
        reference_entities: List of ground truth entities
        predicted_entities: List of predicted entities
    
    Returns:
        Dict with precision, recall, and f1 scores
    """
    ref_set = set(e.lower().strip() for e in reference_entities)
    pred_set = set(e.lower().strip() for e in predicted_entities)
    
    if len(pred_set) == 0 and len(ref_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if len(ref_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    true_positives = len(ref_set & pred_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(ref_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def extract_entities_from_text(text: str, entity_list: Optional[List[str]] = None) -> List[str]:
    """
    Extract entities from generated text.
    
    Args:
        text: Text to extract entities from
        entity_list: Optional list of known entities to match against
    
    Returns:
        List of extracted entities
    """
    if entity_list is None:
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 1] if words else []
        return entities
    
    text_lower = text.lower()
    found_entities = []
    for entity in entity_list:
        if entity.lower() in text_lower:
            found_entities.append(entity)
    
    return found_entities


def evaluate_generation(
    model, 
    tokenizer, 
    dataset, 
    device: str = "cpu", 
    num_samples: int = 5, 
    max_length: int = 128,
    compute_metrics: bool = True
) -> List[Dict]:
    """
    Evaluate model's generation capabilities on a dataset subset.
    
    Args:
        model: Trained NeuroSymbolicLM model
        tokenizer: Tokenizer (T5 tokenizer)
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate
        max_length: Maximum generation length
        compute_metrics: Whether to compute BLEU and entity F1
    
    Returns:
        List of dicts with input, generated, target texts and metrics
    """
    model.eval()
    results = []
    
    # Get samples with responses
    samples_with_responses = [
        s for s in dataset 
        if s.get("should_respond", 0) == 1 and "response" in s and s["response"].strip()
    ]
    eval_samples = samples_with_responses[:num_samples]
    
    if len(eval_samples) == 0:
        print("  No samples with responses found for evaluation")
        return results
    
    with torch.no_grad():
        for sample in eval_samples:
            input_text = sample["text"]
            tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            try:
                # Use T5's built-in generation
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=False
                )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"  Generation error: {e}")
                generated_text = ""
            
            target_text = sample.get("response", "")
            
            result = {
                "input": input_text,
                "generated": generated_text,
                "target": target_text
            }
            
            if compute_metrics and generated_text:
                result["bleu"] = compute_bleu_score(target_text, generated_text)
                
                ref_entities = sample.get("entities", [])
                pred_entities = extract_entities_from_text(generated_text, ref_entities)
                entity_metrics = compute_entity_f1(ref_entities, pred_entities)
                result["entity_precision"] = entity_metrics["precision"]
                result["entity_recall"] = entity_metrics["recall"]
                result["entity_f1"] = entity_metrics["f1"]
            
            results.append(result)
    
    model.train()
    return results


def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute aggregate metrics across all results.
    
    Args:
        results: List of evaluation results
    
    Returns:
        Dict with aggregate metrics
    """
    if len(results) == 0:
        return {}
    
    metrics = {}
    
    bleu_scores = [r["bleu"] for r in results if "bleu" in r]
    if bleu_scores:
        metrics["avg_bleu"] = sum(bleu_scores) / len(bleu_scores)
    
    entity_f1 = [r["entity_f1"] for r in results if "entity_f1" in r]
    if entity_f1:
        metrics["avg_entity_f1"] = sum(entity_f1) / len(entity_f1)
    
    entity_precision = [r["entity_precision"] for r in results if "entity_precision" in r]
    if entity_precision:
        metrics["avg_entity_precision"] = sum(entity_precision) / len(entity_precision)
    
    entity_recall = [r["entity_recall"] for r in results if "entity_recall" in r]
    if entity_recall:
        metrics["avg_entity_recall"] = sum(entity_recall) / len(entity_recall)
    
    return metrics


def print_generation_results(results: List[Dict], num_to_print: int = 3, show_metrics: bool = True):
    """Print generation evaluation results."""
    if len(results) == 0:
        return
    
    print(f"\n  Generation Evaluation (showing {min(num_to_print, len(results))} samples):")
    print("  " + "-" * 58)
    
    for i, result in enumerate(results[:num_to_print]):
        print(f"  Sample {i+1}:")
        print(f"    Input:     {result['input'][:60]}...")
        print(f"    Target:    {result['target'][:60]}...")
        print(f"    Generated: {result['generated'][:60]}...")
        
        if show_metrics and "bleu" in result:
            print(f"    BLEU: {result['bleu']:.4f}, Entity F1: {result.get('entity_f1', 0.0):.4f}")
        print()
    
    if show_metrics:
        agg_metrics = compute_aggregate_metrics(results)
        if agg_metrics:
            print("  " + "-" * 58)
            print("  Aggregate Metrics:")
            for name, value in agg_metrics.items():
                print(f"    {name}: {value:.4f}")


def generate_response(
    model, 
    tokenizer, 
    input_text: str, 
    device: str = "cpu", 
    max_length: int = 128,
    temperature: float = 1.0, 
    do_sample: bool = False, 
    top_k: Optional[int] = None, 
    top_p: Optional[float] = None
) -> str:
    """
    Generate a response for a given input text.
    
    Args:
        model: Trained NeuroSymbolicLM model
        tokenizer: Tokenizer (T5 tokenizer)
        input_text: Input text to generate response for
        device: Device to run generation on
        max_length: Maximum generation length
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated response text
    """
    model.eval()
    
    tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k,
            top_p=top_p
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text
