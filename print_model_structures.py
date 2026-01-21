#!/usr/bin/env python3
"""
Script to print hidden sizes of Qwen and Llama models.
This helps determine the number of prompts needed for the attack (# prompts > max(hidden_size)).
"""

from transformers import AutoConfig

# Model references
qwen_refs = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
llama_refs = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]

print("=" * 80)
print("MODEL HIDDEN SIZES")
print("=" * 80)
print()

# Process Qwen models
print("QWEN MODELS:")
print("-" * 80)
for model_name in qwen_refs:
    print(f"Loading config for {model_name}...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        print(f"  → Hidden Size: {hidden_size}")
        print()
    except Exception as e:
        print(f"  → Error: {e}")
        print()

# Process Llama models
print("LLAMA MODELS:")
print("-" * 80)
for model_name in llama_refs:
    print(f"Loading config for {model_name}...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        print(f"  → Hidden Size: {hidden_size}")
        print()
    except Exception as e:
        print(f"  → Error: {e}")
        print()

# Find maximum hidden size
print("=" * 80)
print("RECOMMENDATION:")
print("-" * 80)

all_models = qwen_refs + llama_refs
max_hidden_size = 0

for model_name in all_models:
    try:
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'hidden_size') and config.hidden_size > max_hidden_size:
            max_hidden_size = config.hidden_size
    except:
        pass

if max_hidden_size > 0:
    recommended_prompts = max_hidden_size + 500  # Add buffer
    print(f"Maximum hidden size across all models: {max_hidden_size}")
    print(f"Recommended number of prompts: {recommended_prompts}")
    print(f"  (max_hidden_size + buffer to ensure coverage)")
else:
    print("Could not determine maximum hidden size")

print("=" * 80)
