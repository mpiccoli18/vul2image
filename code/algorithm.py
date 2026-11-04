import subprocess
import os
import re
import shutil
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
                    
        # Input is a list of tuples (u, v, weight)
        if isinstance(dependencies, list):
            for item in dependencies:
                if len(item) == 3:
                    u, v, weight = item
                    # Convert 1-based Joern line numbers to 0-based Python indices
                    u_idx = int(u) - 1
                    v_idx = int(v) - 1
                    if 0 <= u_idx < n and 0 <= v_idx < n:
                        matrix[u_idx, v_idx] = float(weight) * 255.0
                        matrix[v_idx, u_idx] = float(weight) * 255.0
                        
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

def parseDotFiles(dot):
    """
    Parses all Joern-exported .dot files in a directory to extract line-to-line dependency edges (line_i, line_y, weight)

    :param dot: the path of the directory containing the .dot files
    :return: the edges of each graph
    """
    edges = []

    if not os.path.exists(dot):
        print(f"[-] Error: {dot} not found.")
        return edges

    for filename in os.listdir(dot):
         if filename.endswith(".dot"):
                filepath = os.path.join(dot, filename)
                node_lines = {}

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                   content = f.read()

                node_matches = re.findall(r'"([^"]+)"\s*\[(.*?)\]', content, re.DOTALL)
                for node_id, attrs in node_matches:
                        node_id = node_id.strip()
                        line_match = re.search(r'(?:,\s*|)(\d+)<BR/>', attrs)
                        if line_match:
                            node_lines[node_id] = int(line_match.group(1))
                    
                #Extract edges connecting nodes: "node1" -> "node2"
                edge_matches = re.findall(r'"([^"]+)"\s*->\s*"([^"]+)"', content)
                for src, dst in edge_matches:
                        src = src.strip()
                        dst = dst.strip()
                        
                        # Map node IDs to actual source code line numbers
                        if src in node_lines and dst in node_lines:
                            edges.append((node_lines[src], node_lines[dst], 1.0))
                        
    return edges


def joernPipeline(source):
    """
    Automatically triggers Joern commands from inside Python.
    This helps us parsing the file for extracting CDG and DDG of the source code
    
    :param source: the source file for detection
    """
    for export_dir in ["export_cdg", "export_ddg"]:
        if os.path.exists(export_dir):
            try:
                shutil.rmtree(export_dir)
                print(f"[*] Cleared old directory: {export_dir}")
            except Exception as e:
                print(f"[-] Warning: Could not delete {export_dir}: {e}")
    
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
    parse = subprocess.run([JOERN_EXPORT, "cpg.bin", "--repr", "cdg", "--format", "dot", "--out", "export_cdg"], capture_output=True, text=True, shell=True)
    if parse.stdout:
            print("[Joern STDOUT]:", parse.stdout)
    if parse.stderr:
            print("[Joern STDERR]:", parse.stderr)
            
    if parse.returncode != 0:
            print(f"[-] Joern parsing failed with exit code {parse.returncode}")
            return False
    
    # Export Data Dependency Graph (DDG)
    parse = subprocess.run([JOERN_EXPORT, "cpg.bin", "--repr", "ddg", "--format", "dot", "--out", "export_ddg"], capture_output=True, text=True, shell=True)

    if parse.stdout:
            print("[Joern STDOUT]:", parse.stdout)
    if parse.stderr:
            print("[Joern STDERR]:", parse.stderr)
            
    if parse.returncode != 0:
            print(f"[-] Joern parsing failed with exit code {parse.returncode}")
            return False
    
    print("[+] Joern export complete!")
    return True

if __name__ == "__main__":
    file = "test/vulnerable.c"

    if not os.path.exists(file):
        print(f"[-] Error: {file} not found.")
        exit(1)

    success = joernPipeline(file)

    if success:
        #Read the file
        lines = read_sourceFile(file)
        cdg_edges = parseDotFiles("export_cdg")
        ddg_edges = parseDotFiles("export_ddg")
        print(f"[+] Extracted {len(cdg_edges)} Control Dependency edges (CDG).")
        print(f"[+] Extracted {len(ddg_edges)} Data Dependency edges (DDG).")
        
        converter = Vul2ImageConverter(max_lines=64)
        rgbImage = converter.convert(lines, cdg_edges, ddg_edges)

        print("--- Automated Vul2Image Pipeline Finished ---")
        print(f"Final Image Tensor Shape (C, H, W): {rgbImage.shape}")
        
        # Save output for CNN
        np.save("vulnerabilitesImg.npy", rgbImage)
        