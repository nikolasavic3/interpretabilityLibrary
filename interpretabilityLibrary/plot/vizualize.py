import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import matplotlib.cm as cm

from ..core.base import Explanation

def vizualize(explanation, original_image=None, cmap='jet', title=None, save_path=None):
    """Visualize an explanation result and optionally save to file."""
    attributions = explanation.attributions
    
    # If original image not provided, use the input from explanation
    if original_image is None:
        original_image = explanation.inputs[0].copy()
    
    # Ensure original image is in [0, 1] for plotting
    if original_image.max() > 1.0:
        original_image = original_image / 255.0
    
    # Get attribution map and properly handle shapes
    attr_map = attributions[0]  # Take first sample
    attr_map = np.squeeze(attr_map)  # Remove dimensions of size 1
    
    # Normalize the attribution map to [0, 1] for visualization
    attr_min, attr_max = attr_map.min(), attr_map.max()
    if attr_max > attr_min:
        attr_map = (attr_map - attr_min) / (attr_max - attr_min)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Plot original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot attribution heatmap
    im = axes[1].imshow(attr_map, cmap=cmap)
    axes[1].set_title("Attribution Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot overlay
    axes[2].imshow(original_image)
    heatmap = plt.cm.get_cmap(cmap)(attr_map)
    heatmap[..., 3] = 0.6  # Set alpha
    axes[2].imshow(heatmap)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # For interactive terminals, this will display the plot
    plt.show()