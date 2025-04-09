import matplotlib.pyplot as plt
import numpy as np

def plot_logit_preds_topk(logits_by_layer, save_path, top_k=5):

    layers = [x[0] for x in logits_by_layer]
    all_tokens = [x[1] for x in logits_by_layer]  # list of lists
    all_probs = [x[2] for x in logits_by_layer]   # list of lists

    plt.figure(figsize=(16, 6))
    bar_width = 0.15
    offsets = np.linspace(-bar_width * (top_k-1)/2, bar_width * (top_k-1)/2, top_k)

    for i, (layer, tokens, probs) in enumerate(logits_by_layer):
        for j in range(min(top_k, len(tokens))):
            plt.bar(layer + offsets[j], probs[j], width=bar_width, label=tokens[j] if i == 0 else "", alpha=0.7)
            plt.text(layer + offsets[j], probs[j] + 0.01, tokens[j], ha="center", va="bottom", fontsize=8, rotation=90)

    plt.xlabel("Transformer Layer")
    plt.ylabel("Top-k Token Probabilities")
    plt.title("Top-k Token Predictions per Layer (Logit Lens)")
    plt.xticks(layers, [f"L{i}" for i in layers])
    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()