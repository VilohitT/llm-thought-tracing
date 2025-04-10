from transformer_lens import HookedTransformer, utils
import torch
import torch.nn.functional as F
from properties import device


def run_logit_lens(model: HookedTransformer, prompt: str):
    preds = []
    #Step 1 - Run model on prompt to get cached states
    logits, cache = model.run_with_cache(prompt)
    #Step 2 - Tokenize prompt
    tokenized_prompt = model.to_str_tokens(prompt)
    last_pos_token = len(tokenized_prompt) - 1
    #Step 3 - For each layer, get logit of last token
    for layer in range(model.cfg.n_layers):
      residual_l = cache[f"blocks.{layer}.hook_resid_post"][0, last_pos_token, :]
      logits_layer = residual_l @ model.W_U
      #Step 4 - Using the largest logit for each token, decode the token in thought of the model
      probs = torch.softmax(logits_layer, dim = 0)
      top_k = torch.topk(probs, k=5)
      top_k_indices = top_k.indices.tolist()
      top_k_probs = top_k.values.tolist()
      top_k_tokens = [model.to_single_str_token(token_id) for token_id in top_k_indices]
      preds.append((layer, top_k_tokens, top_k_probs))
    return preds


def cosine_similarity_logits(model: HookedTransformer, prompt: str):
    import gc
    import torch

    tokenized_prompt = model.to_str_tokens(prompt)
    grid_1 = torch.zeros(model.cfg.n_layers, len(tokenized_prompt), device)
    grid_2 = torch.zeros(model.cfg.n_layers, len(tokenized_prompt), device)

    thought_1 = model.to_single_token("Texas")
    thought_2 = model.to_single_token("Austin")
    thought_1_vector = model.W_E[thought_1]
    thought_2_vector = model.W_E[thought_2]

    logits, cache = model.run_with_cache(prompt)

    for layer in range(model.cfg.n_layers):
        for token_pos in range(len(tokenized_prompt)):
            residual_l = cache["resid_post", layer][0, token_pos, :]
            sim_thought_1 = torch.cosine_similarity(residual_l, thought_1_vector, dim=0)
            sim_thought_2 = torch.cosine_similarity(residual_l, thought_2_vector, dim=0)
            grid_1[layer, token_pos] = sim_thought_1.item()
            grid_2[layer, token_pos] = sim_thought_2.item()

        # Optional: empty cache layer-by-layer
        torch.cuda.empty_cache()
        gc.collect()

    # Cleanup after full loop
    del logits, cache, residual_l
    torch.cuda.empty_cache()
    gc.collect()

    return grid_1, grid_2
