from properties import device

def patch_layer_token_residuals(activations, hook, corrupted_cache, layer, pos):
    activations[0, pos, :] = corrupted_cache[layer][0, pos, :]
    return activations

def logit_difference_corrupt(model, prompt, corrupted_city):
    from functools import partial
    import torch, gc

    prompt_tokens = model.to_str_tokens(prompt)
    patching_effect = torch.zeros(model.cfg.n_layers+1, len(prompt_tokens), device= device)

    clean_prompt = 'Dallas is in the state called'
    _, clean_cache = model.run_with_cache(clean_prompt)
    corrupt_prompt = prompt.replace('Dallas', corrupted_city)
    _, corrupt_cache = model.run_with_cache(corrupt_prompt)

    layers = ["blocks.0.hook_resid_pre", *[f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]]
    corrupt_token_id = model.to_single_token("Phoenix")
    orig_token_id = model.to_single_token("Austin")

    for l, layer in enumerate(layers):
        for pos in range(len(prompt_tokens)):
            fwd_hooks = [(layer, partial(patch_layer_token_residuals, corrupted_cache=corrupt_cache, layer=layer, pos=pos))]
            logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)[0, -1]
            diff = logits[corrupt_token_id] - logits[orig_token_id]
            patching_effect[l, pos] = diff.item()  # make it a float

            # Clean up after each run
            del logits, fwd_hooks
            torch.cuda.empty_cache()
            gc.collect()

    del corrupt_cache
    torch.cuda.empty_cache()
    gc.collect()

    return patching_effect



