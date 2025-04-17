import argparse
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


try:
    from data.clean import run_cleaning_and_split
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure your scripts are correctly placed in the 'src' directory and paths are correct.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SentiSynth Project Main Entry Point")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- Clean and Split Command ---
    parser_process = subparsers.add_parser('process_data', help='Clean raw data and create final train/val/sanity splits')
    parser_process.add_argument('--raw-path', default='./data/raw', help='Path to the raw dataset directory')
    parser_process.add_argument('--output-path', default='./data/sst2_dd', help='Path to save the final DatasetDict')
    parser_process.set_defaults(func=lambda args: run_cleaning_and_split(args.raw_path, args.output_path))

    # --- Add other commands here as subparsers ---
    # Example: Download command
    # parser_download = subparsers.add_parser('download', help='Download the raw dataset')
    # parser_download.add_argument('--save-path', default='./data/raw', help='Path to save the raw dataset')
    # parser_download.set_defaults(func=lambda args: run_download(args.save_path)) # Assuming you create run_download

    # Example: Train command
    # parser_train = subparsers.add_parser('train', help='Train a model')
    # parser_train.add_argument('--config', required=True, help='Path to the training configuration file')
    # ... other training args ...
    # parser_train.set_defaults(func=lambda args: run_training(args)) # Assuming you create run_training

    # Parse arguments
    args = parser.parse_args()

    # Execute the function associated with the chosen command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no command is given, print help
        parser.print_help()

if __name__ == "__main__":
    main()
