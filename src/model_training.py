"""
Module for fine-tuning language models with LoRA adaptation.
"""

import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
from accelerate import Accelerator


def load_tokenized_dataset(dataset_dir="data/tokenized"):
    """
    Load tokenized dataset from disk.
    
    Args:
        dataset_dir (str): Directory with tokenized dataset
        
    Returns:
        DatasetDict: Dataset for training
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}. Run dataset_preparation.py first.")
    
    dataset = load_from_disk(dataset_dir)
    print(f"Loaded dataset with {len(dataset['train'])} training and {len(dataset['validation'])} validation examples")
    return dataset


def setup_model_and_tokenizer(model_name="gpt2", use_4bit=True):
    
    """
    Set up the model and tokenizer for fine-tuning.
    
    Args:
        model_name (str): Base model name
        use_4bit (bool): Whether to use 4-bit quantization
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Check for Metal (MPS) availability on M1/M2 Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization if requested
    if use_4bit:
        print(f"Loading {model_name} with 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"Loading {model_name} in full precision")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
        
    # Now we can safely print the model modules
    print("Available model modules:")
    for name, module in model.named_modules():
        if "c_attn" in name or "c_proj" in name:
            print(f"- {name}")
    
    
    return model, tokenizer


def setup_peft_config(lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """
    Set up LoRA configuration for model fine-tuning.
    
    Args:
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout
        
    Returns:
        LoraConfig: LoRA configuration
    """
    # Update target modules to match GPT-2's architecture
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["c_attn", "c_proj"],  # Correct target modules for GPT-2
        bias="none",
    )
    
    print(f"Set up LoRA with rank={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    return peft_config


def train_model(
    model,
    tokenizer,
    dataset,
    output_dir="models/fine_tuned",
    batch_size=8,
    learning_rate=2e-4,
    num_epochs=3,
    gradient_accumulation_steps=4
):
    """
    Train the model using the provided dataset.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        dataset: Dataset for training
        output_dir (str): Output directory
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        num_epochs (int): Number of epochs
        gradient_accumulation_steps (int): Gradient accumulation steps
        
    Returns:
        Trainer: Trained model and trainer
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),  # Use FP16 if on CUDA
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    
    print(f"Starting training for {num_epochs} epochs with learning rate {learning_rate}")
    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Finished training. Model saved to {output_dir}")
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a language model with LoRA')
    parser.add_argument('--dataset', default='data/tokenized', help='Path to tokenized dataset')
    parser.add_argument('--model', default='gpt2', help='Base model name')
    parser.add_argument('--output', default='models/fine_tuned', help='Output directory for fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_tokenized_dataset(args.dataset)
    
    print(f"Setting up model {args.model} and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args.model, args.use_4bit)
    
    print("Setting up LoRA configuration...")
    peft_config = setup_peft_config(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    print("Applying LoRA adapters to model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print(f"Training model and saving to {args.output}...")
    train_model(
        model,
        tokenizer,
        dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main() 