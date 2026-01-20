from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random
import os

def steal_layer(model, full_matrix):
    print("FULL LAYER EXTRACTION ATTACK")

    print("Performing full SVD to recover singular vectors...")
    U, S, Vh = np.linalg.svd(full_matrix, full_matrices=False)

    if hasattr(model.config, 'hidden_size'):
        true_dim = model.config.hidden_size
        print(f"True hidden dimension: {true_dim}")

    W_rec = Vh[:true_dim, :].T @ np.diag(S[:true_dim])
    print(f"Reconstructed W shape: {W_rec.shape}")

    W_true = model.lm_head.weight.detach().numpy()
    print(f"True W shape: {W_true.shape}")

    print("Solving Least Squares to find transformation G...")
    G, residuals, rank, s = np.linalg.lstsq(W_rec, W_true, rcond=None)

    W_aligned = W_rec @ G # compute stolen weight matrix

    # compare stolen to true weights
    diff = W_aligned - W_true

    mse = np.mean(diff ** 2)
    rms = np.sqrt(mse)

    print(f"\nOptimization Complete.")
    print(f"Root Mean Square (RMS) Error: {rms:.2e}")

    if rms < 1e-3:
        print("SUCCESS: Extraction is highly accurate (matches paper results).")
    else:
        print("WARNING: Extraction error is higher than expected.")

    return W_aligned, (mse, rms)

def save_weights(model, weights, error, path):
    os.makedirs(path, exist_ok=True)

    ckpt_path = os.path.join(path, "lm_head_stolen.pt")

    stolen_tensor = torch.from_numpy(weights).float().cpu()

    torch.save({"weights":stolen_tensor,
                "mse": error[0],
                "rms": error[1],
                "hidden_size": model.config.hidden_size,
                "vocab_size": model.config.vocab_size},
               ckpt_path)

    print(f"\nStolen lm_head saved to: {ckpt_path}")

def load_stolen_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location = 'cpu')

    weights = ckpt["weights"]

    device = model.lm_head.weight.device
    model.lm_head.weight.data.copy_(weights.to(device))

    return model


if __name__ == "__main__":
    # recover model
    model_name = "gpt2"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Successfully Loaded {model_name}")
    model.eval()

    vocab_size = tokenizer.vocab_size
    if hasattr(model.config, 'hidden_size'):
        true_dim = model.config.hidden_size
        print(f"True hidden dimension: {true_dim}")

    # generate prompts
    max_queries = 2048
    print(f"\nGenerating {max_queries} unique random prompts...")

    random_prompts = set()
    while len(random_prompts) < max_queries:
        num_tokens = 1
        token_ids = random.sample(range(vocab_size), num_tokens)
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if prompt:
            random_prompts.add(prompt)

    random_prompts = list(random_prompts)
    print(f"Generated {len(random_prompts)} unique prompts")

    # compute logits matrix
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

    full_matrix = np.array(all_logits)
    print(f"Full logits collected. Shape: {full_matrix.shape}")

    # steal layer
    stolen_weights, error = steal_layer(model, full_matrix)

    # save stolen layer weights in file
    save_weights(model, stolen_weights, error, "/scratch/salmanhu/extraction_results/")