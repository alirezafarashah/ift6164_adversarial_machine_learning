#!/usr/bin/env python3
"""
Script to perform scaling family attacks on LLM models.
Collects logits from different model families and saves them to disk.
"""

import argparse
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model family definitions
QWEN_REFS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
LLAMA_REFS = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]


def generate_prompts(tokenizer, model, max_prompts):
    """
    Generate random prompts for querying the model.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        max_prompts: Maximum number of prompts to generate
    
    Returns:
        List of unique random prompts
    """
    vocab_size = tokenizer.vocab_size
    
    if hasattr(model.config, 'hidden_size'):
        true_dim = model.config.hidden_size
        print(f"True hidden dimension: {true_dim}")
    
    print(f"\nGenerating {max_prompts} unique random prompts...")
    
    random_prompts = set()
    while len(random_prompts) < max_prompts:
        num_tokens = 1
        token_ids = random.sample(range(vocab_size), num_tokens)
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if prompt:
            random_prompts.add(prompt)
    
    random_prompts = list(random_prompts)
    print(f"Generated {len(random_prompts)} unique prompts")
    
    return random_prompts


def attack_family(model_names, max_prompts):
    """
    Attack a family of models by collecting their logits.
    
    Args:
        model_names: List of model names to attack
        max_prompts: Number of prompts to use for each model
    """
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        # Load model and tokenizer
        print(f"Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Generate prompts
        random_prompts = generate_prompts(tokenizer, model, max_prompts)
        
        # Collect logits
        print("\nCollecting logits from model...")
        all_logits = []
        all_hidden_states = []
        
        for i, prompt in enumerate(random_prompts):
            if i % 100 == 0:
                print(f"  Query {i}/{len(random_prompts)}")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                final_hidden_state = outputs.hidden_states[-1][0, -1, :].numpy()
                all_hidden_states.append(final_hidden_state)
                
                # Extract last token logits
                logits = outputs.logits[0, -1, :].numpy()
                all_logits.append(logits)
        
        # Convert to numpy array
        full_matrix = np.array(all_logits)
        print(f"Full logits collected. Shape: {full_matrix.shape}")
        
        # Save logits to file
        model_name_clean = model_name.replace('/', '_')
        logits_filename = f"logits_{model_name_clean}.npy"
        np.save(logits_filename, full_matrix)
        print(f"Logits saved to {logits_filename}")
        
        # Save hidden states as well
        hidden_states_matrix = np.array(all_hidden_states)
        hidden_filename = f"hidden_{model_name_clean}.npy"
        np.save(hidden_filename, hidden_states_matrix)
        print(f"Hidden states saved to {hidden_filename}")


def compute_and_save_singular_values(logits_filename, output_filename=None):
    """
    Load logits from a file, compute singular values via SVD, and save them.
    
    Args:
        logits_filename: Path to the .npy file containing logits
        output_filename: Path to save the singular values (default: auto-generated)
    
    Returns:
        Singular values array
    """
    print(f"\nLoading logits from {logits_filename}...")
    logits_matrix = np.load(logits_filename)
    print(f"Logits matrix shape: {logits_matrix.shape}")
    
    # Compute singular values using SVD
    print("Computing singular values via SVD...")
    S = np.linalg.svd(logits_matrix, full_matrices=False, compute_uv=False)
    
    print(f"Number of singular values: {len(S)}")
    print(f"Largest singular value: {S[0]:.2f}")
    if len(S) > 1:
        print(f"Second largest singular value: {S[1]:.2f}")
    print(f"Smallest singular value: {S[-1]:.2e}")
    
    # Auto-generate output filename if not provided
    if output_filename is None:
        base_name = logits_filename.replace('.npy', '')
        output_filename = f"{base_name}_singular_values.npy"
    
    # Save singular values
    np.save(output_filename, S)
    print(f"Singular values saved to {output_filename}")
    
    return S


def analyze_all_logits(pattern="logits_*.npy"):
    """
    Find all logit files matching the pattern and compute their singular values.
    
    Args:
        pattern: Glob pattern to match logit files
    """
    import glob
    
    logit_files = glob.glob(pattern)
    
    if not logit_files:
        print(f"No logit files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(logit_files)} logit file(s) to process")
    
    for logit_file in sorted(logit_files):
        print("\n" + "="*60)
        compute_and_save_singular_values(logit_file)
    
    print("\n" + "="*60)
    print("All singular values computed and saved!")
    print("="*60)


def main():
    """Main function to run the attack based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform scaling family attacks on LLM models"
    )
    parser.add_argument(
        "family",
        type=str,
        choices=["Qwen", "Llama", "analyze"],
        help="Model family to attack (Qwen or Llama) or 'analyze' to process existing logit files"
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=2048,
        help="Maximum number of prompts to generate (default: 2048)"
    )
    parser.add_argument(
        "--logit-file",
        type=str,
        help="Specific logit file to analyze (only used with 'analyze' command)"
    )
    
    args = parser.parse_args()
    
    # Handle analyze command
    if args.family == "analyze":
        if args.logit_file:
            # Analyze a specific file
            compute_and_save_singular_values(args.logit_file)
        else:
            # Analyze all logit files in current directory
            analyze_all_logits()
        return
    
    # Select the appropriate model family
    if args.family == "Qwen":
        model_refs = QWEN_REFS
        print("Attacking Qwen family models...")
    elif args.family == "Llama":
        model_refs = LLAMA_REFS
        print("Attacking Llama family models...")
    else:
        raise ValueError(f"Unknown family: {args.family}")
    
    # Run the attack
    attack_family(model_refs, args.max_prompts)
    
    print("\n" + "="*60)
    print("Attack complete!")
    print("="*60)



if __name__ == "__main__":
    main()
