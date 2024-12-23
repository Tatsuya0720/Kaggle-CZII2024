import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def visualize_apo_ferritin(tomogram, coords, n_slices=3, slice_thickness=10):
    """
    Visualize apo-ferritin particles in tomogram slices.
    """
    fig = plt.figure(figsize=(20, 10))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, n_slices),
                    axes_pad=0.3,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.1)
    
    # Normalize tomogram data
    vmin, vmax = np.percentile(tomogram, (1, 99))
    normalized_tomogram = np.clip((tomogram - vmin) / (vmax - vmin), 0, 1)
    
    # Calculate evenly spaced z-positions
    z_positions = np.linspace(0, tomogram.shape[0]-1, n_slices, dtype=int)
    
    # Plot each slice
    for idx, ax in enumerate(grid):
        z = z_positions[idx]
        
        # Show tomogram slice
        im = ax.imshow(normalized_tomogram[z, :, :], cmap='gray', vmin=0, vmax=1)
        
        # Find particles near this slice
        mask = np.abs(coords[:, 0] - z) < slice_thickness
        if np.any(mask):
            ax.scatter(coords[mask, 2], coords[mask, 1],
                      color='red', marker='o', s=100, 
                      facecolors='none', linewidth=2,
                      label='apo-ferritin')
        
        ax.set_title(f'Slice Z={z}\n({np.sum(mask)} particles visible)')
        ax.grid(False)
        
        # Set the axes limits to match the tomogram dimensions
        ax.set_xlim(0, tomogram.shape[2])
        ax.set_ylim(tomogram.shape[1], 0)  # Inverted y-axis to match image coordinates
    
    # Add colorbar and title
    grid.cbar_axes[0].colorbar(im)
    
    plt.suptitle('Apo-ferritin Particles in Tomogram Slices\n' + 
                 f'Showing particles within ±{slice_thickness} units of each slice',
                 fontsize=16, y=1.05)
    
    # Add legend to the first subplot
    grid[0].legend(bbox_to_anchor=(1.5, 1.0))
    
    plt.show()
    