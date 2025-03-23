"""
Module for preparing text datasets for transformer models.
"""

import os
import argparse
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd


def load_processed_chunks(filepath='data/processed/all_chunks.txt'):
    """
    Load processed text chunks from file.
    
    Args:
        filepath (str): Path to the processed chunks file
        
    Returns:
        list: List of text chunks
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}. Run data_cleaning.py first.")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = f.read().split('\n\n')
    
    print(f"Loaded {len(chunks)} text chunks from {filepath}")
    return chunks


def create_dataset(chunks, model_name="gpt2"):
    """
    Create a Hugging Face dataset from text chunks.
    
    Args:
        chunks (list): List of text chunks
        model_name (str): Model name for tokenizer
        
    Returns:
        DatasetDict: Dataset for training and evaluation
    """
    # Create a dataframe from chunks
    df = pd.DataFrame({"text": chunks})
    
    # Create a dataset from the dataframe
    dataset = Dataset.from_pandas(df)
    
    # Handle case when we have very few samples
    if len(chunks) < 3:  # Need at least 3 samples to have 1 in validation with 10% split
        # Use train_test_split but rename the "test" split to "validation"
        temp_split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            'train': temp_split['train'],
            'validation': temp_split['test']  # Rename 'test' to 'validation'
        })
    else:
        # Split into train and validation sets (90/10 split)
        temp_split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            'train': temp_split['train'],
            'validation': temp_split['test']  # Rename 'test' to 'validation'
        })
    
    print(f"Created dataset with {len(dataset['train'])} training and {len(dataset['validation'])} validation examples")
    return dataset


def tokenize_dataset(dataset, model_name="gpt2", max_length=512):
    """
    Tokenize the dataset for model training.
    
    Args:
        dataset (DatasetDict): Dataset to tokenize
        model_name (str): Model name for tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        DatasetDict: Tokenized dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    print(f"Tokenized dataset using {model_name} tokenizer")
    return tokenized_dataset, tokenizer


def save_dataset(tokenized_dataset, output_dir="data/tokenized"):
    """
    Save the tokenized dataset to disk.
    
    Args:
        tokenized_dataset (DatasetDict): Tokenized dataset
        output_dir (str): Output directory
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataset
    tokenized_dataset.save_to_disk(output_dir)
    print(f"Saved tokenized dataset to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for transformer model')
    parser.add_argument('--input', default='data/processed/all_chunks.txt', help='Input file with processed chunks')
    parser.add_argument('--output', default='data/tokenized', help='Output directory for tokenized dataset')
    parser.add_argument('--model', default='gpt2', help='Model name for tokenizer')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print(f"Loading chunks from {args.input}...")
    chunks = load_processed_chunks(args.input)
    
    print(f"Creating dataset...")
    dataset = create_dataset(chunks, args.model)
    
    print(f"Tokenizing dataset with {args.model} tokenizer...")
    tokenized_dataset, _ = tokenize_dataset(dataset, args.model, args.max_length)
    
    print(f"Saving dataset to {args.output}...")
    save_dataset(tokenized_dataset, args.output)
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main() 