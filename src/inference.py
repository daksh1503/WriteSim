"""
Module for generating text using the fine-tuned model.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_and_tokenizer(model_dir="models/fine_tuned", base_model="gpt2"):
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_dir (str): Directory with fine-tuned model
        base_model (str): Base model name
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Check if model exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model not found at {model_dir}. Run model_training.py first.")
    
    # Check if this is a regular model or a PEFT model
    is_peft_model = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    # Check for Metal (MPS) availability on M1/M2 Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load model
    if is_peft_model:
        print(f"Loading PEFT model from {model_dir} based on {base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            torch_dtype=torch.float16,
            device_map={"": device}
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        print(f"Loading regular model from {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map={"": device}
        )
    
    model.eval()
    return model, tokenizer


def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_length=100, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=50,
    num_return_sequences=1
):
    """
    Generate text from a prompt.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt (str): Prompt text
        max_length (int): Maximum length of generated text
        temperature (float): Temperature for generation
        top_p (float): Top-p sampling parameter
        top_k (int): Top-k sampling parameter
        num_return_sequences (int): Number of sequences to generate
        
    Returns:
        list: Generated texts
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up the generated text
    generated_texts = []
    for seq in output:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def interactive_mode(model, tokenizer):
    """
    Run an interactive text generation session.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        
    Returns:
        None
    """
    print("\n===== Interactive Text Generation =====")
    print("Enter a prompt to generate text. Type 'exit' to quit.")
    print("You can customize generation with parameters like:")
    print("  --length=100 --temp=0.7 --top_p=0.9 --top_k=50 --num=1")
    
    while True:
        # Get user input
        user_input = input("\nPrompt> ")
        if user_input.lower() == "exit":
            break
        
        # Parse generation parameters
        max_length = 100
        temperature = 0.7
        top_p = 0.9
        top_k = 50
        num_sequences = 1
        
        # Check for parameter overrides
        parts = user_input.split()
        prompt_parts = []
        
        for part in parts:
            if part.startswith("--length="):
                try:
                    max_length = int(part.split("=")[1])
                except ValueError:
                    pass
            elif part.startswith("--temp="):
                try:
                    temperature = float(part.split("=")[1])
                except ValueError:
                    pass
            elif part.startswith("--top_p="):
                try:
                    top_p = float(part.split("=")[1])
                except ValueError:
                    pass
            elif part.startswith("--top_k="):
                try:
                    top_k = int(part.split("=")[1])
                except ValueError:
                    pass
            elif part.startswith("--num="):
                try:
                    num_sequences = int(part.split("=")[1])
                except ValueError:
                    pass
            else:
                prompt_parts.append(part)
        
        # Reconstruct prompt without parameters
        prompt = " ".join(prompt_parts)
        
        # Generate text
        print(f"\nGenerating {num_sequences} response(s) with: length={max_length}, temp={temperature}, top_p={top_p}, top_k={top_k}")
        generated_texts = generate_text(
            model,
            tokenizer, 
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_sequences
        )
        
        # Print generated text
        for i, text in enumerate(generated_texts):
            print(f"\n--- Response {i+1} ---")
            print(text)


def main():
    parser = argparse.ArgumentParser(description='Generate text using a fine-tuned model')
    parser.add_argument('--model_dir', default='models/fine_tuned', help='Directory with fine-tuned model')
    parser.add_argument('--base_model', default='gpt2', help='Base model name (for PEFT models)')
    parser.add_argument('--prompt', default='', help='Prompt for text generation (if empty, runs in interactive mode)')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--num_sequences', type=int, default=1, help='Number of sequences to generate')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model)
    
    if not args.prompt:
        # Run in interactive mode
        interactive_mode(model, tokenizer)
    else:
        # Generate text from command line argument
        print(f"Generating text from prompt: {args.prompt}")
        generated_texts = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_sequences
        )
        
        for i, text in enumerate(generated_texts):
            print(f"\n--- Response {i+1} ---")
            print(text)


if __name__ == "__main__":
    main() 