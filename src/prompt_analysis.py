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

    def _perform_causal_tracing(self, prompt, target_tokens, target_indices, results, replacement_map, corruption_strategy = "map"):
        """
            Perform causal tracing by corrupting the input and analyzing how
            the corruption propagates through the model. (Layer-wise)
        """

        # Get final token position
        final_pos = len(self.model.to_str_tokens(prompt)) - 1

        # Create corrupted prompt based on the replacement map
        if corruption_strategy == "map":
            corrupted_prompt = prompt
            for original, replacement in replacement_map.items():
                corrupted_prompt = corrupted_prompt.replace(original, replacement)
        else:
            raise ValueError("Unsupported corruption strategy. Use 'map'.")

        # Run both clean and corrupted prompts
        clean_logits, clean_cache = model.run_with_cache(prompt)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupted_prompt)

        # Compute effect of patching from corrupted to clean
        patching_effects = {}

        def patching_hook(activations, hook):
            # Replace corrupted activation with clean activation
            activations[0, final_pos, :] = clean_cache[hook.name][0, final_pos, :]
            return activations
        
        for token, token_idx in zip(target_tokens, target_indices):
            layer_effects = []
            
            for layer in range(model.cfg.n_layers):
                # Compute corrupted logits with patching
                hook_point = f"blocks.{layer}.hook_resid_post"
                patched_logits = model.run_with_hooks(
                    corrupted_prompt,
                    fwd_hooks=[(hook_point, patching_hook)]
                )

                # Measure effect on target token probability
                clean_logit = clean_logits[0, final_pos, token_idx].item()
                corrupt_logit = corrupt_logits[0, final_pos, token_idx].item()
                patched_logit = patched_logits[0, final_pos, token_idx].item()

                # Effect is how much patching recovers the clean prediction
                if corrupt_logit != clean_logit:  # Avoid division by zero
                    effect = (patched_logit - corrupt_logit) / (clean_logit - corrupt_logit)
                else:
                    effect = 0.0

                layer_effects.append((layer, effect))
            
            patching_effects[token] = layer_effects
        results["causal_tracing"] = patching_effects
                
        # Clean up
        del clean_cache, corrupt_cache
        torch.cuda.empty_cache()
        gc.collect()

    def _analyze_token_influence(self, prompt, target_tokens, results):         ## This function is not refined yet
        """
            Analyze how each input token influences the logits for the target tokens.
        """
        tokens = model.to_str_tokens(prompt)
        n_tokens = len(tokens)

        # For each input token, mask it and see how it affects the output
        token_influences = {token: [] for token in target_tokens}

        for mask_pos in range(n_tokens - 1):  # Don't mask the final token
            # Replace token with a neutral token (like [PAD] or similar)
            masked_tokens = model.to_tokens(prompt).clone()
            masked_tokens[0, mask_pos] = model.tokenizer.encode(" ")[0]

            # Run the model with the masked input
            masked_logits = model(masked_tokens)

            # Compare with normal logits for each target token
            normal_logits, _ = model.run_with_cache(prompt)

            final_pos = n_tokens - 1

            for token in target_tokens:
                token_idx = model.to_single_token(token)
                masked_logit = masked_logits[0, final_pos, token_idx].item()
                normal_logit = normal_logits[0, final_pos, token_idx].item()

                # Influence is the difference in logits
                influence = normal_logit - masked_logit
                token_influences[token].append((mask_pos, tokens[mask_pos], influence))

        # Sort influences by magnitude
        for token in target_tokens:
            token_influences[token] = sorted(
                token_influences[token],
                key=lambda x: abs(x[2]),
                reverse=True
            )

        results["token_influences"] = token_influences

    def run_logit_lens_all_positions(self, prompt, top_k=5, include_probs=True):
        """
        Run logit lens to analyze what the model "wants to predict" at every position and layer.
        
        This enhanced version of logit lens examines all token positions, not just the final token,
        giving a comprehensive view of how representations evolve through the network.
        
        Args:
            prompt: Text input to analyze
            top_k: Number of top predictions to return for each position and layer
            include_probs: Whether to include probability scores with predictions
            
        Returns:
            Dictionary containing the full analysis results
        """
        # Run model with activation caching
        tokens = self.model.to_tokens(prompt)
        tokenized_prompt = self.model.to_str_tokens(prompt)
        logits, cache = self.model.run_with_cache(prompt)
        
        # Initialize results structure
        results = {
            "tokens": tokenized_prompt,
            "layers": [],
        }
        
        # For each layer, analyze all token positions
        for layer in range(self.n_layers):
            layer_data = {
                "layer_num": layer,
                "positions": []
            }
            
            # For each token position
            for pos in range(len(tokenized_prompt)):
                # Get residual stream at this position and layer
                residual = cache[f"blocks.{layer}.hook_resid_post"][0, pos, :]
                
                # Project to get "logits" at this intermediate state
                intermediate_logits = residual @ self.model.W_U
                
                # Get top-k predictions
                if include_probs:
                    probs = torch.softmax(intermediate_logits, dim=0)
                    top_values, top_indices = torch.topk(probs, k=top_k)
                    top_probs = top_values.cpu().tolist()
                else:
                    top_values, top_indices = torch.topk(intermediate_logits, k=top_k)
                    top_probs = None
                
                # Convert token indices to strings
                top_tokens = [self.model.to_single_str_token(idx.item()) for idx in top_indices]
                
                # Store position data
                pos_data = {
                    "position": pos,
                    "token": tokenized_prompt[pos],
                    "top_tokens": top_tokens,
                    "top_probs": top_probs
                }
                
                layer_data["positions"].append(pos_data)
            
            results["layers"].append(layer_data)

        results["token_tracking"] = self._track_tokens_across_layers(results, tokens_to_track=None)
    
        return results

    def _track_tokens_across_layers(self, logit_lens_results, tokens_to_track=None):
        """
        Extract the probability of specific tokens across all layers and positions.
        This helps track how concepts evolve through the network.
        
        Args:
            logit_lens_results: Results from run_logit_lens_all_positions
            tokens_to_track: List of specific tokens to track (if None, use top tokens from last layer)
            
        Returns:
            Dictionary mapping tracked tokens to their probabilities across layers and positions
        """
        all_tokens = logit_lens_results["tokens"]
        
        # If no specific tokens are provided, use top tokens from last layer's last position
        if tokens_to_track is None:
            last_layer = logit_lens_results["layers"][-1]
            last_pos = last_layer["positions"][-1]
            tokens_to_track = last_pos["top_tokens"][:3]  # Track top 3 tokens by default
        
        # Initialize tracking dictionary
        tracking = {token: [] for token in tokens_to_track}
        
        # For each token to track, extract its probability across all layers
        for track_token in tokens_to_track:
            # Initialize grid for this token [n_layers, n_positions]
            grid = torch.zeros(self.n_layers, len(all_tokens))
            
            # Fill in probabilities where available
            for layer_idx, layer_data in enumerate(logit_lens_results["layers"]):
                for pos_idx, pos_data in enumerate(layer_data["positions"]):
                    if track_token in pos_data["top_tokens"]:
                        token_idx = pos_data["top_tokens"].index(track_token)
                        grid[layer_idx, pos_idx] = pos_data["top_probs"][token_idx]
            
            tracking[track_token] = grid
        
        return tracking

    


    