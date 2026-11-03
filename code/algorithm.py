import subprocess
import os
import numpy as np
import pandas as pd

JOERN_BIN_DIR = "C:\\Program Files (x86)\\joern-cli"
JOERN_PARSE = os.path.join(JOERN_BIN_DIR, "joern-parse.bat")
JOERN_EXPORT = os.path.join(JOERN_BIN_DIR, "joern-export.bat")

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

    def _dependency_to_matrix(self, dependencies, lines):
        """
        Helper to build n x n adjacency matrices for Control or Data dependencies.
        :param dependencies: List of tuples representing edges (line_i, line_j, weight)
        """
        matrix = np.zeros((self.max_lines, self.max_lines), dtype=np.float32)
        n = min(len(lines), self.max_lines)
        
        # Input is a Pandas DataFrame from Joern CSV
        if isinstance(dependencies, pd.DataFrame) and not dependencies.empty:
            for _, row in dependencies.iterrows():
                try:
                    u = int(row.get('startLine', row.get('source_line', 0))) - 1
                    v = int(row.get('endLine', row.get('target_line', 0))) - 1
                    weight = float(row.get('weight', 1.0))
                    if 0 <= u < n and 0 <= v < n:
                        matrix[u, v] = weight * 255.0
                        matrix[v, u] = weight * 255.0
                except (ValueError, TypeError):
                    continue
                    
        # Input is a list of tuples (u, v, weight)
        elif isinstance(dependencies, list):
            for item in dependencies:
                if len(item) == 3:
                    u, v, weight = item
                    if 0 <= u < n and 0 <= v < n:
                        matrix[u, v] = float(weight) * 255.0
                        matrix[v, u] = float(weight) * 255.0
                        
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
        red_channel = self._dependency_to_matrix(control_dependencies, code_lines)

        # Green Channel: Data Dependency Graph (DDG) Matrix
        green_channel = self._dependency_to_matrix(data_dependencies, code_lines)

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
    cmd = [JOERN_PARSE, source, "-o", "cpg.bin"]
    parse = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    if parse.stdout:
        print("[Joern STDOUT]:", parse.stdout)
    if parse.stderr:
        print("[Joern STDERR]:", parse.stderr)
        
    if parse.returncode != 0:
        print(f"[-] Joern parsing failed with exit code {parse.returncode}")
        return False

    print("[+] Parsing complete. CPG generated successfully.")
    
    # Export Control Dependency Graph (CDG)
    subprocess.run([JOERN_EXPORT, "--repr", "cdg", "--format", "csv", "--out", "export_cdg"], capture_output=True, text=True, shell=True)
    
    # Export Data Dependency Graph (DDG)
    subprocess.run([JOERN_EXPORT, "--repr", "ddg", "--format", "csv", "--out", "export_ddg"], capture_output=True, text=True, shell=True)
    
    print("[+] Joern export complete!")
    return True

if __name__ == "__main__":
    file = "vul2image/test/vulnerable.c"

    if not os.path.exists(file):
        print(f"[-] Error: {file} not found.")
        exit(1)

    success = joernPipeline(file)

    if success:
        #Read the file
        lines = read_sourceFile(file)
        cdg_path = os.path.join("export_cdg", "edges.csv")
        ddg_path = os.path.join("export_ddg", "edges.csv")
        try:
            # Extract CDG and DDG from joern csv files
            cdg = pd.read_csv(cdg_path) if os.path.exists(cdg_path) else pd.DataFrame(columns=['startLine', 'endLine', 'weight'])
            ddg = pd.read_csv(ddg_path) if os.path.exists(ddg_path) else pd.DataFrame(columns=['startLine', 'endLine', 'weight'])

        except FileNotFoundError:
            print("[-] Dependency CSVs not found yet. Make sure your export step generated them.")
            cdg = pd.DataFrame()
            ddg = pd.DataFrame()
        
        converter = Vul2ImageConverter(max_lines=64)
        rgbImage = converter.convert(lines, cdg, ddg)

        print("--- Automated Vul2Image Pipeline Finished ---")
        print(f"Final Image Tensor Shape (C, H, W): {rgbImage.shape}")
        
        # Save output for CNN
        np.save("vulnerabilitesImg.npy", rgbImage)