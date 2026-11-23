import os
import numpy as np 
import pandas as pd
import gdown
import zipfile
import shutil
import time
from multiprocessing import Pool
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
        # on_bad_lines='skip' ensures a malformed row doesn't crash the script
        df = pd.read_csv(csv, engine="python", on_bad_lines='skip')
        
        print(f"[+] Successfully loaded {len(df)} rows.")
        print("[+] Columns found in dataset:", df.columns.tolist())
        return df
        
    except Exception as e:
        print(f"[-] Failed to read CSV: {e}")
        return None


def process_sample(args):
    
    
    idx, func_before, label_idx, outputNpyDir, baseDir, workers = args
    workerId = idx % workers
    workDir = os.path.join(baseDir, f"worker_{workerId}")
    os.makedirs(workDir, exist_ok=True)

    tmpFile = os.path.join(workDir, f"tmp_sample_{idx}.c")
    cgdDir = os.path.join(workDir, "export_cdg")
    ddgDir = os.path.join(workDir, "export_ddg")

    try:
        with open(tmpFile, "w", encoding="utf-8") as f:
            f.write(func_before)
        
        success = joernPipeline(tmpFile, workDir=workDir)
        if not success:
            return False

        converter = Vul2ImageConverter(max_lines=64)
        lines = read_sourceFile(tmpFile)
        cdg = parseDotFiles(cdg_dir)
        ddg = parseDotFiles(ddg_dir)

        rgbImage = converter.convert(lines, cdg, ddg)
        npyName = f"bigVul_idx_{idx}_vul_{label_idx}.npy"
        np.save(os.path.join(output_npy_dir, npyName), rgbImage)
        return True

    except Exception as e:
        print(f"[-] Error on index {idx}: {e}")
        return False

    finally:
        if os.path.exists(tmpFile):
            try:
                os.remove(tmpFile)
            except OSError:
                pass

def process_dataset(df, outputNpyDir, max_samples=20):

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

    dfVul = df[pd.to_numeric(df['vul'], errors='coerce') == 1.0]
    print(f"[+] Found {len(dfVul)} vulnerable samples out of {len(df)} total rows.")
    
    valid = df[
        df["func_before"].notna() &
        df["func_before"].apply(lambda s: isinstance(s, str) and len(s.strip()) > 0)
    ].copy()
    valid["label_idx"] = pd.to_numeric(valid["vul"], errors="coerce").fillna(0).astype(int)
    valid["label_idx"] = (valid["label_idx"] == 1).astype(int)

    df0 = valid[valid["label_idx"] == 0]
    df1 = valid[valid["label_idx"] == 1]

    samples = max(1, max_samples // 2)
    noVul = min(samples, len(df0))
    vul = min(samples, len(df1))

    sub0 = df0.sample(n=noVul, random_state=42)
    sub1 = df1.sample(n=vul, random_state=42)

    converter = Vul2ImageConverter(max_lines=64)

    dfSub = (
        pd.concat([sub0, sub1])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    
    print(f"[+] Selected {len(dfSub)} balanced samples ({noVul} safe, {vul} vulnerable).")
    
    workers = 4
    tmpDir = "/tmp/workers"
    os.makedirs(tmpDir, exist_ok=True)

    tasks = [
        (
            int(row.name),
            row["func_before"],
            int(row["label_idx"]),
            outputNpyDir,
            tmpDir,
            workers
        )
        for _, row in dfSub.iterrows()
    ]

    start_t = time.time()
    with Pool(processes=workers) as pool:
        results = pool.map(process_sample, tasks)
    
    elapsed = time.time() - start_t
    print(f"[+] Completed {sum(results)}/{len(results)} samples in {elapsed:.1f}s ({workers} workers).")
    
    '''
    for index, row in dfSub.iterrows():
        source = row['func_before']
        label = row['vul']

        # Writing the dataframe's string code into a tmp file
        tmp_filename = f"tmp_sample_{index}.c"
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

            label_idx = int(float(label)) if not pd.isna(label) else 1
            npyName = "bigVul_" + f"idx_{index}_vul_{label_idx}.npy"
            
            np.save(os.path.join(outputNpyDir, npyName), rgbImage)

            print(f"[+] Saved tensor: {npyName}")

        except Exception as e:
            print(f"[-] Error on index {index}: {e}")
        
        finally:
            # Clean up the temporary .c file from disk
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
    '''
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

    # File path where the dataset is located
    DATASET_ROOT = "MSR_data_cleaned.csv"
    
    # Folder where all .npy tensors are stored
    OUTPUT_NPY_DIR = "output/dataset_output"
    
    #download_dataset()             # Only if you don't have downloaded yet BigVul
    
    df = load_dataset(DATASET_ROOT)
    
    if df is not None:
        process_dataset(df, OUTPUT_NPY_DIR)
    else:
        print("[-] Aborting process because the dataset failed to load.")