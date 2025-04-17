import re
from datasets import load_from_disk, disable_caching, DatasetDict
import os

# Disable caching to ensure fresh processing if script is rerun
disable_caching()

def normalize(txt):
    if txt is None:
        return ""
    txt = txt.lower().strip()
    return " ".join(txt.split())

def run_cleaning_and_split(raw_data_path="./data/raw", output_dd_path="./data/sst2_dd"):
    """Loads raw data, normalizes text, splits train/sanity, and saves the final DatasetDict."""
    print("--- Running Data Cleaning & Splitting ---")
    print(f"Loading raw dataset from {raw_data_path}...")

    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data directory not found at {raw_data_path}")
        print("Please ensure the raw data has been downloaded or saved correctly.")
        return

    try:
        ds_raw = load_from_disk(raw_data_path)
        print("Raw dataset loaded.")
    except Exception as e:
        print(f"Error loading dataset from disk: {e}")
        return

    print("Applying normalization...")
    num_processors = os.cpu_count() or 1
    print(f"Using {num_processors} processes for mapping.")

    try:
        # Create a 'text' column with normalized sentences
        ds_clean = ds_raw.map(
            lambda ex: {"text": normalize(ex["sentence"])},
            num_proc=num_processors,
            desc="Normalizing text"
        )
        print("Normalization complete.")
    except Exception as e:
        print(f"Error during mapping/normalization: {e}")
        return

    print("Splitting training data to create 'sent_sanity' (5% holdout)...")
    try:
        train_full = ds_clean["train"]
        val = ds_clean["validation"] # Keep original validation split
        test = ds_clean["test"]

        # Split the cleaned training data
        split_result = train_full.train_test_split(test_size=0.05, seed=42, shuffle=True)
        train = split_result['train']
        sent_sanity = split_result['test']

        print(f"Final split sizes -> Train: {len(train)}, Val: {len(val)}, Sanity: {len(sent_sanity)}")

        # Create the final DatasetDict
        final_dd = DatasetDict({
            "train": train,
            "val": val,
            "sent_sanity": sent_sanity,
            "test": test
        })
        print("Final DatasetDict created:")
        print(final_dd)

    except Exception as e:
        print(f"Error during data splitting: {e}")
        return

    print(f"Saving final DatasetDict to {output_dd_path}...")
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_dd_path), exist_ok=True)
        final_dd.save_to_disk(output_dd_path)
        print("Final DatasetDict saved successfully.")
    except Exception as e:
        print(f"Error saving final DatasetDict: {e}")
        return

    print("--- Data Cleaning & Splitting Finished ---")


if __name__ == "__main__":
    # Default paths if run directly
    default_raw_path = "./data/raw"
    default_output_path = "./data/sst2_dd"
    run_cleaning_and_split(raw_data_path=default_raw_path, output_dd_path=default_output_path) 
    