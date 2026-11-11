import os
import numpy as np 
import shutil
from algorithm import read_sourceFile, parseDotFiles, Vul2ImageConverter, joernPipeline

def process_dataset(datasetRoot, outputNpyDir, max_lines=64):

    """
    Recursively walks through the nested Juliet C test cases directory,
    runs the Joern pipeline on every .c file, and saves unique .npy tensors.
    """

    
    if os.path.exists(outputNpyDir):
        try:
            shutil.rmtree(outputNpyDir)
            print(f"[*] Cleared old directory: {outputNpyDir}")
            os.makedirs(outputNpyDir)
        except Exception as e:
            print(f"[-] Warning: Could not delete {outputNpyDir}: {e}")
    else:
        os.makedirs(outputNpyDir)
    
    converter = Vul2ImageConverter(max_lines=max_lines)

    success_idx = 0
    fail = 0

    print(f"[*] Scanning dataset directory: {datasetRoot}")

    for root, dirs, files in os.walk(datasetRoot):
        for file in files:
            if file.endswith(".c") or file.endswith(".cpp"):
                cFile = os.path.join(root, file)

                relativePath = os.path.relpath(cFile, datasetRoot)
                if file.endswith(".c"):
                    uniquePre = relativePath.replace(os.sep, "_").replace(".c", "")
                else:
                    uniquePre = relativePath.replace(os.sep, "_").replace(".cpp", "")

                print(f"\n----------------------------------------")
                print(f"Processing: {relativePath}")

                try:
                    success = joernPipeline(cFile)
                    if not success:
                        print(f"[-] Joern failed for {relativePath}")
                        fail += 1
                        continue

                    lines = read_sourceFile(cFile)
                    cdg = parseDotFiles("export_cdg")
                    ddg = parseDotFiles("export_ddg")

                    rgb_image = converter.convert(lines, cdg, ddg)
                    print(f"[+] Extracted -> CDG: {len(cdg)} edges, DDG: {len(ddg)} edges")

                    npyFile = f"{uniquePre}.npy"
                    npyPath = os.path.join(outputNpyDir, npyFile)

                    np.save(npyPath, rgb_image)
                    print(f"[+] Saved tensor shape {rgb_image.shape} to: {npyPath}")
                    success_idx += 1

                except Exception as e:
                    print(f"[-] Error processing {relativePath}: {e}")
                    fail += 1

    print("\n========================================")
    print(" Dataset Batch Processing Summary")
    print("========================================")
    print(f"Successfully Processed: {success}")
    print(f"Failed / Skipped:       {fail}")


if __name__ == "__main__":
    DATASET_ROOT = "test/C/testcases/CWE15_External_Control_of_System_or_Configuration_Setting"
    
    # Folder where all .npy tensors are stored
    OUTPUT_NPY_DIR = "test/dataset_output"
    
    process_dataset(DATASET_ROOT, OUTPUT_NPY_DIR, max_lines=64)