from transformer_lens import HookedTransformer, utils
import torch
import torch.nn.functional as F


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