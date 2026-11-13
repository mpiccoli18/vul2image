import os
import numpy as np 
import pandas as pd
import gdown
import zipfile
import shutil
from algorithm import read_sourceFile, parseDotFiles, Vul2ImageConverter, joernPipeline


def load_dataset(csv):
    """
    Safely loads the BigVul dataset using Pandas
    """

    if not os.path.exists(csv):
        print(f"[-] Error: {csv} not found. Run your download script first!")
        return None

    print("[*] Loading Big-Vul dataset into Pandas...")

    try:
        # engine='python' safely handles complex multi-line strings and messy quotes in source code
        # on_bad_lines='skip' ensures a malformed row doesn't crash your entire script
        df = pd.read_csv(csv, engine="python", on_bad_lines='skip')
        
        print(f"[+] Successfully loaded {len(df)} rows.")
        print("[+] Columns found in dataset:", df.columns.tolist())
        return df
        
    except Exception as e:
        print(f"[-] Failed to read CSV: {e}")
        return None



def process_dataset(df, outputNpyDir, max_samples=100):

    """
    Processes the entire csv dataset file by looping inside with row and columns.
    When a source code is found, it saves the code inside a temponary C file.
    Afterwards, the tmp file gets sent to the Joern pipeline for exploiting the vulnerabilities
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

    dfSub = df.head(max_samples)
    converter = Vul2ImageConverter(max_lines=64)

    for index, row in dfSub.iterrows():
        source = row['func_before']
        label = row['vul']

        if pd.isna(source) or not isinstance(source, str):
            continue

        # Writing the dataframe's string code into a tmp file
        tmp_filename = f"tmp_sample_{index}"
        with open(tmp_filename, "w", encoding="utf-8") as f:
            f.write(source)


        print(f"\n----------------------------------------")
        print(f"Processing Big-Vul Index {index} (Label: {label})")

        try: 
            success = joernPipeline(tmp_filename)
            if not success:
                print(f"[-] Joern failed for sample {index}")
                continue

            lines = read_sourceFile(tmp_filename)
            cdg = parseDotFiles("export_cdg")
            ddg = parseDotFiles("export_ddg")

            rgbImage = converter.convert(lines, cdg, ddg)

            label = 1 if label == "1" else "0"
            npyName = "bigVul_" + f"idx_{index}_vul_{label}.npy"
            np.save(os.path.join(outputNpyDir, npyName), rgbImage)

            print(f"[+] Saved tensor: {npyName}")

        except Exception as e:
            print(f"[-] Error on index {index}: {e}")
        
        finally:
            # Clean up the temporary .c file from disk
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

def download_dataset():
    """
    Download and extract the BigVul dataset if it's not present inside the repo
    """
    csv = "MSR_data_cleaned.csv"
    zip = "MSR_data_cleaned.zip"
    fileID = "1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X"
    url = f"https://drive.google.com/uc?id={fileID}"

    if not os.path.isfile(csv):
        print("[*] Downloading Big-Vul dataset...")
        gdown.download(url, zip, quiet=False)

        if os.path.exists(zip):
            print("[*] Extracting archive...")
            with zipfile.ZipFile(zip, 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove(zip)
            print("[+] Big-Vul dataset ready!")
        else:
            print("[-] Error: Download failed or zip file not found.")
    else:
        print("Already downloaded bigvul data")

if __name__ == "__main__":

    DATASET_ROOT = "MSR_data_cleaned.csv"
    # Folder where all .npy tensors are stored
    OUTPUT_NPY_DIR = "test/dataset_output"
    
    download_dataset()
    
    df = load_dataset(DATASET_ROOT)
    
    if df is not None:
        process_dataset(df, OUTPUT_NPY_DIR)
    else:
        print("[-] Aborting process because the dataset failed to load.")
    
    process_dataset(df, OUTPUT_NPY_DIR)