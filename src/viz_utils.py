import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def plot_patching_effect_grid(patching_effect, prompt_tokens, save_path="patching_effect_grid.png"):
    # Convert tensor to numpy
    effect = patching_effect.cpu().numpy()

    # Create diverging colormap: blue → white → red
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # blue to red

    plt.figure(figsize=(len(prompt_tokens) * 0.6, effect.shape[0] * 0.4))
    ax = sns.heatmap(
        effect,
        cmap=cmap,
        center=0.0,
        xticklabels=prompt_tokens,
        yticklabels=[f"L{l}" for l in range(effect.shape[0])],
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Logit Difference (Corrupt - Original)'}
    )

    plt.title("Patching Effect: Token vs. Layer")
    plt.xlabel("Token Position")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_thought_similarity_grids(grid_1, grid_2, prompt_tokens, save_path="thought_trace.png"):
    grid_1 = grid_1.cpu().numpy()
    grid_2 = grid_2.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(len(prompt_tokens) * 1.2, 8))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    layer_labels = [f"L{i}" for i in range(grid_1.shape[0])]

    sns.heatmap(grid_1, cmap=cmap, center=0, xticklabels=prompt_tokens, yticklabels=layer_labels,
                cbar_kws={"label": "Cosine Similarity"}, ax=axes[0])
    axes[0].set_title("Similarity to 'Texas'")
    axes[0].set_xlabel("Token Position")
    axes[0].set_ylabel("Layer")

    sns.heatmap(grid_2, cmap=cmap, center=0, xticklabels=prompt_tokens, yticklabels=layer_labels,
                cbar_kws={"label": "Cosine Similarity"}, ax=axes[1])
    axes[1].set_title("Similarity to 'Austin'")
    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Layer")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()