"""Evaluation utilities for generation quality assessment."""

from typing import List, Dict, Optional
import torch


def evaluate_generation(
    model, 
    tokenizer, 
    dataset, 
    device: str = "cpu", 
    num_samples: int = 5, 
    max_length: int = 128
) -> List[Dict]:
    """
    Evaluate model's generation capabilities on a dataset subset.
    
    Args:
        model: Trained NeuroSymbolicLM model
        tokenizer: Tokenizer for encoding/decoding
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate
        max_length: Maximum generation length
    
    Returns:
        List of dicts with input, generated, and target texts
    """
    model.eval()
    results = []
    
    # Get samples with responses
    samples_with_responses = [
        s for s in dataset 
        if s.get("should_respond", 0) == 1 and "response" in s
    ]
    eval_samples = samples_with_responses[:num_samples]
    
    if len(eval_samples) == 0:
        print("  No samples with responses found for evaluation")
        return results
    
    # Get special token IDs
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 1
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
    pad_token_id = tokenizer.pad_token_id or 0
    
    with torch.no_grad():
        for sample in eval_samples:
            input_text = sample["text"]
            tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                temperature=1.0,
                do_sample=False
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            target_text = sample.get("response", "")
            
            results.append({
                "input": input_text,
                "generated": generated_text,
                "target": target_text
            })
    
    model.train()
    return results


def print_generation_results(results: List[Dict], num_to_print: int = 3):
    """Print generation evaluation results."""
    if len(results) == 0:
        return
    
    print(f"\n  Generation Evaluation (showing {min(num_to_print, len(results))} samples):")
    print("  " + "-" * 58)
    for i, result in enumerate(results[:num_to_print]):
        print(f"  Sample {i+1}:")
        print(f"    Input:    {result['input'][:60]}...")
        print(f"    Target:   {result['target'][:60]}...")
        print(f"    Generated: {result['generated'][:60]}...")
        print()


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
        tokenizer: Tokenizer for encoding/decoding
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
    
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 1
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
    pad_token_id = tokenizer.pad_token_id or 0
    
    tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text
