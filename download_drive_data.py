import gdown
import os
import sys

def download_dataset():
    url = "https://drive.google.com/drive/folders/18zP4pHt5E6YqA3usey16ETEzKNeAn5X9"
    output_dir = "til26_dataset"
    
    print(f"Downloading dataset from Google Drive: {url}")
    print("This may take a while depending on the dataset size...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # download_folder downloads all contents inside the drive folder
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
        print("\nDownload complete! Dataset saved to:", os.path.abspath(output_dir))
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Please ensure you have internet access and the link is public.")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()
