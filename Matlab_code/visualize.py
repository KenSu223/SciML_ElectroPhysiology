import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Load the .mat file
data = loadmat('2D_data_1.mat')

# Access the variables
variable = data['data']  # Replace 'variable_name' with the actual variable name
print(variable.shape)

# Generate sample data
i = 806
data1 = variable[i, 0, :, :, 0]  # Heatmap data for the first plot
data2 = variable[i, 1, :, :, 0]  # Heatmap data for the second plot

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# First heatmap
heatmap1 = axs[0].imshow(data1, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
axs[0].set_title(f't = {4 * (i % 8 + 1)}ms')
axs[0].set_xlabel('X-axis 1')
axs[0].set_ylabel('Y-axis 1')
fig.colorbar(heatmap1, ax=axs[0])  # Add colorbar to the first subplot

# Second heatmap
heatmap2 = axs[1].imshow(data2, cmap='viridis', vmin=0, vmax=1, aspect='auto')
axs[1].set_title(f't = {4 * (i % 8 + 2)}ms')
axs[1].set_xlabel('X-axis 2')
axs[1].set_ylabel('Y-axis 2')
fig.colorbar(heatmap2, ax=axs[1])  # Add colorbar to the second subplot

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()