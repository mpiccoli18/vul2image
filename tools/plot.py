import numpy as np
import matplotlib.pyplot as plt

tensor = np.load("vulnerabilitesImg.npy")
print(f"Loaded tensor shape: {tensor.shape}")

# Transpose from PyTorch/Channels-First (C, H, W) to Channels-Last (H, W, C)
img_array = np.transpose(tensor, (1, 2, 0))

# Ensure values are strictly within 0-255 and cast to unsigned 8-bit integer
img_array = np.clip(img_array, 0, 255).astype(np.uint8)

# Plot and save the RGB image
plt.figure(figsize=(6, 6))
plt.imshow(img_array)
plt.title("Vul2Image Source Code Representation\n(Red: Control Flow | Green: Data Flow | Blue: Text)", fontsize=10)
plt.axis("off")

plt.savefig("vulnerability_visualization.png", bbox_inches='tight', dpi=300)
print("Saved visualization as 'vulnerability_visualization.png'")

plt.show()