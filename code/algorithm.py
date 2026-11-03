import subprocess
import os
import numpy as np
import pandas as pd

class Vul2ImageConverter:
    def __init__(self, max_lines=64):
        """
        Initializes the Vul2Image converter.
        :param max_lines: Fixed height and width (n x n) for the output matrix 
                          to ensure uniform image sizes for the CNN.
        """
        self.max_lines = max_lines

    def _text_to_feature_matrix(self, code_lines):
        """
        Blue Channel: Code Text / Semantic Feature Matrix.
        Converts lines of source code into numeric character/token ASCII codes 
        and pads/truncates them into an (max_lines x max_lines) matrix.
        """
        matrix = np.zeros((self.max_lines, self.max_lines), dtype=np.float32)
        n = min(len(code_lines), self.max_lines)
        
        for i in range(n):
            line = code_lines[i]
            # Convert characters of the code line to numeric values
            chars = [ord(c) for c in line[:self.max_lines]]
            for j, val in enumerate(chars):
                matrix[i, j] = val
                
        # Normalize pixel values to standard 0-255 image range
        if matrix.max() > 0:
            matrix = (matrix / matrix.max()) * 255.0
            
        return matrix

    def _dependency_to_matrix(self, dependencies):
        """
        Helper to build n x n adjacency matrices for Control or Data dependencies.
        :param dependencies: List of tuples representing edges (line_i, line_j, weight)
        """
        matrix = np.zeros((self.max_lines, self.max_lines), dtype=np.float32)
        
        for u, v, weight in dependencies:
            if u < self.max_lines and v < self.max_lines:
                # Scale connection weight to 255
                matrix[u, v] = float(weight) * 255.0
                matrix[v, u] = float(weight) * 255.0  # Symmetric mapping
                
        return matrix

    def convert(self, code_lines, control_dependencies, data_dependencies):
        """
        Main function to convert a source code snippet into a 3-channel RGB image tensor.
        
        :param code_lines: List of strings (lines of source code).
        :param control_dependencies: Edges representing control flow (e.g., if/loops). Format: [(u, v, weight), ...]
        :param data_dependencies: Edges representing data flow (e.g., variable usage). Format: [(u, v, weight), ...]
        :return: Numpy array of shape (3, max_lines, max_lines) representing the RGB image.
        """
        # Red Channel: Control Dependency Graph (CDG) Matrix
        red_channel = self._dependency_to_matrix(control_dependencies)

        # Green Channel: Data Dependency Graph (DDG) Matrix
        green_channel = self._dependency_to_matrix(data_dependencies)

        # Blue Channel: Code Text / Semantic Feature Matrix
        blue_channel = self._text_to_feature_matrix(code_lines)

        # Stack the three matrices into a 3-channel RGB image tensor
        rgb_tensor = np.stack([red_channel, green_channel, blue_channel], axis=0)

        return rgb_tensor

def read_sourceFile(file_path):
    """
    Reads a source code file from disk line by line
    
    :param file_path: the path of the C/C++ source file
    :return: Lines of code without whitespace or newlines
    """
    with open(file_path, 'r', encoding = 'utf-8', errors='ignore') as f:
        code_lines = [line.rstrip('\n').rstrip('\r') for line in f.readlines()]
    return code_lines

def joernPipeline(source):
    """
    Automatically triggers Joern commands from inside Python.
    This helps us parsing the file for extracting CDG and DDG of the source code
    
    :param source: the source file for detection
    """
    parse = subprocess.run(["joern-parse", source], capture_output=True, text=True)

    if parse != 0:
        print("[-] Error running joern-parse:", parse.stderr)
        return False

    print("[+] Parsing complete. CPG generated successfully.")
    return True

if __name__ == "__main__":
    file = "vulnerable_sample.c"

    if not os.path.exists(file):
        print(f"[-] Error: {file} not found.")
        exit(1)

    success = joernPipeline(file)

    if success:
        #Read the file
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip('\n').rstrip('\r') for line in f.readlines()]
        try:
            # Extract CDG and DDG from joern csv files
            cdg = pd.read_csv("joern_cdg_edges.csv")
            ddg = pd.read_csv("joern_cdg_edges.csv")

        except FileNotFoundError:
            print("[-] Dependency CSVs not found yet. Make sure your export step generated them.")
            cdg_df = pd.DataFrame(columns=['source_line', 'target_line', 'weight'])
            ddg_df = pd.DataFrame(columns=['source_line', 'target_line', 'weight'])
        
        converter = Vul2ImageConverter(max_lines=64)
        rgbImage = converter.convert(lines, cdg, ddg)

        print("--- Automated Vul2Image Pipeline Finished ---")
        print(f"Final Image Tensor Shape (C, H, W): {rgbImage.shape}")
        
        # Save output for CNN model
        np.save("vulnerabilitesImg.npy", rgbImage)