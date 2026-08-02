"""
Download APTOS 2019 dataset using Kaggle API
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import subprocess

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_aptos():
    print("📥 Downloading APTOS 2019 Blindness Detection dataset...")
    
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    
    print("Authenticated with Kaggle.")
    
    competition = "aptos2019-blindness-detection"
    
    # Download
    api.competition_download_files(competition, path=str(DATA_DIR), quiet=False)
    
    zip_path = DATA_DIR / f"{competition}.zip"
    
    if zip_path.exists():
        print(f"✅ Downloaded to {zip_path}")
        print("Unzipping...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("✅ Unzipped successfully!")
        
        # Clean zip
        zip_path.unlink()
        
        print("\nDataset ready at:", DATA_DIR)
        print("Expected structure:")
        print("  data/raw/train_images/*.png")
        print("  data/raw/train.csv")
    else:
        print("❌ Download failed. Please check your Kaggle credentials.")
        print("1. Create ~/.kaggle/kaggle.json")
        print("2. Accept competition rules on Kaggle website.")

if __name__ == "__main__":
    download_aptos()