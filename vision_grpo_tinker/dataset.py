import pandas as pd
import torch
import io
from torch.utils.data import Dataset
from PIL import Image

class Geometry3K_PandasDataset(Dataset):
    def __init__(self, parquet_path):
        """
        Args:
            parquet_path (str): Path to the .parquet file (e.g., 'test.parquet')
        """
        print(f"[Dataset] Loading data from {parquet_path}...")
        
        # 1. Load Dataframe using Pandas
        self.df = pd.read_parquet(parquet_path)
        
        # 2. Filter: Remove rows with missing answers or images
        # We check if 'answer' is not null and not empty string
        initial_len = len(self.df)
        self.df = self.df[self.df['answer'].notna() & (self.df['answer'] != "")]
        self.df = self.df[self.df['images'].notna()]
        
        print(f"[Dataset] Loaded {len(self.df)} valid samples (filtered from {initial_len}).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row as a Series
        row = self.df.iloc[idx]
        
        # --- A. Image Extraction ---
        # The 'images' column is typically a list of dicts: [{'bytes': b'...', 'path': ...}]
        try:
            image_data = row['images'][0] # Get first image object
            
            if isinstance(image_data, dict) and 'bytes' in image_data:
                # Standard HF Parquet format
                image_bytes = image_data['bytes']
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else:
                # Fallback if it's raw bytes or different format
                image = Image.new('RGB', (224, 224), color='black')
                print(f"Warning: Could not parse image for index {idx}")
                
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        # --- B. Text Cleaning & Prompt Formatting ---
        # Raw text: "<image>Find x."
        # We strip "<image>" because we add our own special tokens later.
        raw_problem = str(row['problem']).replace("<image>", "").strip()
        
        # Construct the "Visual R1" Prompt
        # Force the model to start with a thought block
        formatted_prompt = (
            f"User: <|image_pad|> {raw_problem}\n"
            f"Assistant: <think>"
        )

        # --- C. Ground Truth ---
        # We pass the raw answer string (e.g., "2 \sqrt{221}")
        # The rewards.py orchestrator will handle the math parsing.
        ground_truth = str(row['answer']).strip()

        return {
            "image": image,
            "prompt": formatted_prompt,
            "ground_truth": ground_truth,
            "id": idx
        }

def visual_r1_collate_fn(batch):
    """
    Custom collate function for VLM training.
    Returns lists of images/texts instead of stacked tensors, 
    because Tinker/Qwen handles the tokenization internally.
    """
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    ids = [item['id'] for item in batch]
    
    return images, prompts, ground_truths, ids

# --- Quick Test Block ---
if __name__ == "__main__":
    import urllib.request
    import os
    
    PARQUET_PATH = "train-00000-of-00001.parquet"
    
    # Download if not exists
    if not os.path.exists(PARQUET_PATH):
        print("Downloading parquet file...")
        url = "https://huggingface.co/datasets/hiyouga/geometry3k/resolve/main/data/train-00000-of-00001.parquet"
        urllib.request.urlretrieve(url, PARQUET_PATH)
        print("Download complete!")
    
    ds = Geometry3K_PandasDataset(PARQUET_PATH)
    sample = ds[0]
    
    print("\n--- Sample 0 ---")
    print(f"Prompt:\n{sample['prompt']}")
    print(f"Ground Truth: {sample['ground_truth']}")
    print(f"Image Size: {sample['image'].size}")