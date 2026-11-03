import numpy as np

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


if __name__ == "__main__":
    #This is only an example of a code snippet, nothing more
    vulnerable_code_snippet = [
        "void process_input(char *input) {",
        "    char buffer[16];",
        "    if (strlen(input) > 16) {",
        "        strcpy(buffer, input);",  # Vulnerability: Potential buffer overflow
        "    }",
        "}"
    ]
    
    # Mock Control Dependencies (CDG)
    control_edges = [
        (2, 3, 1.0),
        (2, 4, 1.0)
    ]
    
    # Mock Data Dependencies (DDG)
    # Variables flowing between lines (e.g., 'input' used in strlen and strcpy)
    data_edges = [
        (0, 2, 1.0),  # 'input' parameter passed into 'if' condition
        (0, 3, 1.0),  # 'input' parameter passed into 'strcpy'
        (1, 3, 1.0)   # 'buffer' declaration interacts with 'strcpy' target
    ]
    
    converter = Vul2ImageConverter(max_lines=32)
    
    # Generate the RGB image tensor
    rgb_image_tensor = converter.convert(
        code_lines=vulnerable_code_snippet,
        control_dependencies=control_edges,
        data_dependencies=data_edges
    )
    
    print("--- Vul2Image Conversion Complete ---")
    print(f"Output Tensor Shape (C, H, W): {rgb_image_tensor.shape}")
    print(f"Red Channel (Control Flow) max value: {rgb_image_tensor[0].max():.2f}")
    print(f"Green Channel (Data Flow) max value: {rgb_image_tensor[1].max():.2f}")
    print(f"Blue Channel (Text Features) max value: {rgb_image_tensor[2].max():.2f}")