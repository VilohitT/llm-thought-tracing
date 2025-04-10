from transformer_lens import HookedTransformer

def load_gpt2_small():
  return HookedTransformer.from_pretrained("gpt2-small", device = "cuda")