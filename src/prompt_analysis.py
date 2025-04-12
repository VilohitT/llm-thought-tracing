class CircuitDiscoverer:

  def __init__(self, model: HookedTransformer):
    self.model = model
    self.n_layers = model.cfg.n_layers
    self.n_heads = model.cfg.n_heads
    self.d_model = model.cfg.d_model
    self.d_mlp = model.cfg.d_mlp if hasattr(model.cfg, 'd_mlp') else 4 * model.cfg.d_model

  def analyse_prompt(self, prompt: str, target_tokens: list, save_intermediates=True):
    results = {
            "prompt": prompt,
            "target_tokens": target_tokens,
            "tokens": self.model.to_str_tokens(prompt),
            "token_ids": self.model.to_tokens(prompt)[0].tolist()
        }
    
    print("Running model with activation caching...")
    # model outputs and cache
    logits, cache = self.model.run_with_cache(prompt)
    results["logits"] = logits.detach().cpu()
    
    # token indices for target tokens
    target_indices = [self.model.to_single_token(token) for token in target_tokens]
    results["target_indices"] = target_indices
    
    # information from cache
    print("Extracting attention patterns...")
    self._extract_attention_patterns(cache, results)

    print("Analyzing component contributions...")
    self._analyze_component_contributions(prompt, target_tokens, target_indices, cache, results)
    
    #add logit lens

  def _extract_attention_patterns(self, cache, results):

    seq_len = cache["blocks.0.attn.hook_pattern"].shape[2]
    attn_scores = torch.zeros(
    (model.cfg.n_layers, model.cfg.n_heads, prompt_length, prompt_length),
    device=device)
    for layer in range(model.cfg.n_layers):
      attn_scores[layer] = cache[f"blocks.{layer}.attn.hook_pattern"][0]
    
    results["attention_patterns"] = attn_scores.detach().cpu()

  def _analyze_component_contributions(self, prompt, target_tokens, target_indices, cache, results):
        """Analyze how each component contributes to target token logits."""
        # Get final token position
        final_pos = len(self.model.to_str_tokens(prompt)) - 1
        
        # Initialize storage for contributions
        contributions = {token: {} for token in target_tokens}
        
        # For each layer
        for layer in range(self.n_layers):
            # Analyze attention heads
            attn_out = cache[f"blocks.{layer}.attn.hook_result"][0, final_pos]
            
            for head in range(self.n_heads):
                head_output = attn_out[head]
                
                # Project through unembedding matrix to get logit contributions
                for idx, token in zip(target_indices, target_tokens):
                    head_contribution = (head_output @ self.model.W_U[idx]).item()
                    contributions[token][f"L{layer}H{head}"] = head_contribution
            
            # Analyze MLP contribution
            mlp_out = cache[f"blocks.{layer}.mlp.hook_result"][0, final_pos]
            for idx, token in zip(target_indices, target_tokens):
                mlp_contribution = (mlp_out @ self.model.W_U[idx]).item()
                contributions[token][f"L{layer}MLP"] = mlp_contribution
        
        # Find top contributors for each token
        top_contributors = {}
        for token, comps in contributions.items():
            sorted_comps = sorted(comps.items(), key=lambda x: abs(x[1]), reverse=True)
            top_contributors[token] = sorted_comps[:10]  # Top 10 contributors
        
        results["component_contributions"] = contributions
        results["top_contributors"] = top_contributors