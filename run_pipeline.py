#!/usr/bin/env python3
"""
Script to run the complete WriteSim pipeline:
1. Clean input text
2. Prepare dataset
3. Train model
4. Launch Gradio interface
"""

import os
import argparse
import subprocess
import sys


def run_command(command, description):
    """
    Run a shell command and print output.
    
    Args:
        command (list): Command to run
        description (str): Description of the command
        
    Returns:
        int: Return code of the command
    """
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    
    try:
        process = subprocess.run(command, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return e.returncode


def check_data_directory():
    """
    Check if data/raw directory has text or PDF files.
    
    Returns:
        bool: True if files exist, False otherwise
    """
    if not os.path.exists("data/raw"):
        print("Error: data/raw directory not found.")
        return False
    
    valid_files = [f for f in os.listdir("data/raw") if f.endswith((".txt", ".pdf"))]
    if not valid_files:
        print("Error: No .txt or .pdf files found in data/raw directory.")
        return False
    
    print(f"Found {len(valid_files)} text or PDF files in data/raw directory.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run the complete WriteSim pipeline')
    parser.add_argument('--model', default='gpt2', help='Base model name')
    parser.add_argument('--skip_cleaning', action='store_true', help='Skip data cleaning step')
    parser.add_argument('--skip_dataset', action='store_true', help='Skip dataset preparation step')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    
    args = parser.parse_args()
    
    # Check data directory if not skipping cleaning
    if not args.skip_cleaning and not check_data_directory():
        print("Please add your text files to the data/raw directory and try again.")
        return 1
    
    # Step 1: Clean data
    if not args.skip_cleaning:
        result = run_command(
            ["python", "src/data_cleaning.py"],
            "Cleaning and processing text data"
        )
        if result != 0:
            print("Error in data cleaning step. Aborting pipeline.")
            return result
    
    # Step 2: Prepare dataset
    if not args.skip_dataset:
        result = run_command(
            ["python", "src/dataset_preparation.py", "--model", args.model],
            "Preparing dataset for training"
        )
        if result != 0:
            print("Error in dataset preparation step. Aborting pipeline.")
            return result
    
    # Step 3: Train model
    if not args.skip_training:
        command = [
            "python", "src/model_training.py",
            "--model", args.model,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lora_r", str(args.lora_r)
        ]
        
        if args.use_4bit:
            command.append("--use_4bit")
        
        result = run_command(
            command,
            "Training model with LoRA"
        )
        if result != 0:
            print("Error in model training step. Aborting pipeline.")
            return result
    
    # Step 4: Launch Gradio interface
    print("\n")
    print("=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)
    print("\nLaunching Gradio interface. Press Ctrl+C to stop.")
    
    try:
        subprocess.run(["python", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nGradio interface stopped.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 