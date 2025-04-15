import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_logit_lens_grid(self, results, save_path="logit_lens_grid.png"):
    """
    Create a visualization of the logit lens results as a grid.
    
    Args:
        results: Results from run_logit_lens_all_positions
        save_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    
    # Extract token strings for labeling
    tokens = results["tokens"]
    n_tokens = len(tokens)
    n_layers = self.n_layers
    
    # Create a figure with subplots - one for each important token to track
    token_tracking = results["token_tracking"]
    n_tracked = len(token_tracking)
    
    fig, axes = plt.subplots(n_tracked, 1, figsize=(n_tokens * 0.8, n_tracked * 4), 
                            squeeze=False)
    
    # For each tracked token, create a heatmap
    for i, (token, grid) in enumerate(token_tracking.items()):
        ax = axes[i, 0]
        
        # Convert from tensor if needed
        if isinstance(grid, torch.Tensor):
            grid = grid.cpu().numpy()
        
        # Create heatmap for this token
        sns.heatmap(grid, cmap="viridis", ax=ax, 
                   xticklabels=tokens, 
                   yticklabels=[f"L{i}" for i in range(n_layers)],
                   cbar_kws={"label": "Probability"})
        
        ax.set_title(f"Activation of '{token}' across layers and positions")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def visualize_token_evolution(results, token_position, save_path="token_evolution.png"):
    """
    Visualize how a specific token's representation evolves through the layers.
    This enhanced version shows the top-k tokens at each layer and their probabilities
    using a stacked horizontal bar chart.
    
    Args:
        results: Results from run_logit_lens_all_positions
        token_position: Position index of the token to analyze
        save_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get original token at this position
    tokens = results["tokens"]
    if token_position >= len(tokens):
        raise ValueError(f"Token position {token_position} out of range (max: {len(tokens)-1})")
    
    original_token = tokens[token_position]
    
    # Extract the top-k predictions at each layer for this position
    top_k = 5  # Show top 5 tokens at each layer
    layers = []
    all_tokens = set()  # Keep track of all unique tokens
    layer_data_list = []
    
    for layer_idx, layer_data in enumerate(results["layers"]):
        pos_data = layer_data["positions"][token_position]
        top_tokens = pos_data["top_tokens"][:top_k]
        top_probs = pos_data["top_probs"][:top_k] if pos_data["top_probs"] else [1.0] + [0.0] * (top_k - 1)
        
        # Save layer data
        layers.append(layer_idx)
        layer_data_list.append((top_tokens, top_probs))
        
        # Update unique tokens
        all_tokens.update(top_tokens)
    
    # Convert to list and sort for consistent coloring
    unique_tokens = sorted(list(all_tokens))
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', len(unique_tokens))
    colors = [cmap(i) for i in range(len(unique_tokens))]
    
    # Create horizontal stacked bar chart for each layer
    for i, (top_tokens, top_probs) in enumerate(layer_data_list):
        left = 0  # Starting position for each bar segment
        
        # Plot each token's probability as a segment
        for token, prob in zip(top_tokens, top_probs):
            plt.barh(i, prob, left=left, color=colors[token_to_idx[token]], alpha=0.7)
            
            # Only add text if probability is significant
            if prob > 0.05:
                # Position text in the middle of the segment
                text_x = left + prob / 2
                plt.text(text_x, i, token, ha='center', va='center', 
                        fontsize=9, color='black', fontweight='bold')
            
            left += prob
    
    # Add a legend for the tokens
    # Only include tokens that have significant probability in at least one layer
    significant_tokens = []
    for token in unique_tokens:
        for top_tokens, top_probs in layer_data_list:
            if token in top_tokens:
                idx = top_tokens.index(token)
                if top_probs[idx] > 0.05:
                    significant_tokens.append(token)
                    break
    
    handles = [plt.Rectangle((0,0), 1, 1, color=colors[token_to_idx[token]]) 
              for token in significant_tokens]
    plt.legend(handles, significant_tokens, loc='upper left', 
              bbox_to_anchor=(1.01, 1), title="Tokens")
    
    # Add labels and title
    plt.xlabel("Probability")
    plt.ylabel("Layer")
    plt.yticks(layers, [f"Layer {l}" for l in layers])
    plt.title(f"Evolution of Token at Position {token_position}: '{original_token}'")
    plt.grid(True, axis='x', alpha=0.3)
    plt.xlim(0, 1.05)
    
    # Highlight the original token's position (e.g. "Dallas")
    plt.axhline(y=-0.5, color='black', linestyle='-', alpha=0.2)
    plt.text(-0.05, -1, f"Input Position {token_position}: '{original_token}'", 
            fontsize=10, fontweight='bold', ha='left')
    
    # Save and return
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path