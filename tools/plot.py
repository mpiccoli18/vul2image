import os
import numpy as np
import shutil
import matplotlib.pyplot as plt


def convertAll(inputNpyDir, outputPlotDir):
    """
    Loops through all .npy tensors in the directory, converts them, 
    and saves them as visual .png image representations.
    """
    os.makedirs(outputPlotDir, exist_ok=True)

    if not os.path.exists(inputNpyDir):
        print(f"[-] Directory '{inputNpyDir}' not found. Make sure you generated your tensors first!")
        return

    if os.path.exists(outputPlotDir):
        try:
            shutil.rmtree(outputPlotDir)
            print(f"[*] Cleared old directory: {outputPlotDir}")
            os.makedirs(outputPlotDir)
        except Exception as e:
            print(f"[-] Warning: Could not delete {outputPlotDir}: {e}")
    else:
        os.makedirs(outputPlotDir)

    npyFiles = [f for f in os.listdir(inputNpyDir) if f.endswith(".npy")]
    print(f"[*] Found {len(npyFiles)} tensor files to plot in '{inputNpyDir}'.")

    for filename in npyFiles:
        filePath = os.path.join(inputNpyDir, filename)

        try:
            tensor = np.load(filePath)
            print(f"Loaded tensor shape: {tensor.shape}")

            # Transpose from PyTorch/Channels-First (C, H, W) to Channels-Last (H, W, C)
            img_array = np.transpose(tensor, (1, 2, 0))

            # Ensure values are strictly within 0-255 and cast to unsigned 8-bit integer
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            base_name = os.path.splitext(filename)[0]
            
            # Plot and save the RGB image
            plt.figure(figsize=(6, 6))
            plt.imshow(img_array)
            plt.title(f"Vul2Image: {base_name}\n(Red: Control Flow | Green: Data Flow | Blue: Text)", fontsize=10)
            plt.axis("off")

            outputPath = os.path.join(outputPlotDir, f"{base_name}.png")
            
            plt.savefig(outputPath, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"[+] Saved plot: {outputPath}")
        
        except Exception as e:
            print(f"[-] Error plotting {filename}: {e}")

    print("\n[+] Batch plotting complete! All images saved to:", outputPlotDir)



if __name__ == "__main__":
    convertAll("test/dataset_output", "test/plot_output")