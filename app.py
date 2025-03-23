"""
Gradio interface for the fine-tuned language model.
"""

import os
import torch
import gradio as gr
from src.inference import load_model_and_tokenizer, generate_text


# Global variables for model and tokenizer
model = None
tokenizer = None


def initialize_model(model_dir, base_model):
    """
    Initialize the model and tokenizer.
    
    Args:
        model_dir (str): Directory with fine-tuned model
        base_model (str): Base model name
        
    Returns:
        tuple: (success, message)
    """
    global model, tokenizer
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_dir, base_model)
        return True, f"Model loaded successfully from {model_dir}"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def generate(
    prompt, 
    model_dir="models/fine_tuned",
    base_model="gpt2",
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_sequences=1
):
    """
    Generate text based on the prompt.
    
    Args:
        prompt (str): Text prompt
        model_dir (str): Directory with fine-tuned model
        base_model (str): Base model name
        max_length (int): Maximum length of generated text
        temperature (float): Temperature for generation
        top_p (float): Top-p sampling parameter
        top_k (int): Top-k sampling parameter
        num_sequences (int): Number of sequences to generate
        
    Returns:
        str: Generated text
    """
    global model, tokenizer
    
    # Initialize model if not already loaded
    if model is None or tokenizer is None:
        success, message = initialize_model(model_dir, base_model)
        if not success:
            return message
    
    # Check if prompt is empty
    if not prompt:
        return "Please enter a prompt to generate text."
    
    try:
        # Generate text
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
        
        # Format the output
        if num_sequences > 1:
            result = ""
            for i, text in enumerate(generated_texts):
                result += f"Response {i+1}:\n{text}\n\n"
            return result.strip()
        else:
            return generated_texts[0]
    
    except Exception as e:
        return f"Error generating text: {str(e)}"


def create_ui():
    """
    Create the Gradio interface.
    
    Returns:
        gr.Interface: Gradio interface
    """
    # Default model paths
    default_model_dir = "models/fine_tuned"
    default_base_model = "gpt2"
    
    # Check if model directory exists
    model_exists = os.path.exists(default_model_dir)
    
    # Create interface
    with gr.Blocks(title="WriteSim: Your Writing Style Generator") as interface:
        gr.Markdown("# WriteSim: Generate Text in Your Writing Style")
        
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a prompt to generate text in your style...",
                    lines=3
                )
                
                generate_button = gr.Button("Generate", variant="primary")
                
                output = gr.Textbox(
                    label="Generated Text",
                    placeholder="Generated text will appear here...",
                    lines=10
                )
            
            with gr.Column(scale=1):
                with gr.Accordion("Advanced Settings", open=False):
                    model_dir = gr.Textbox(
                        label="Model Directory",
                        value=default_model_dir,
                        placeholder="Path to fine-tuned model"
                    )
                    
                    base_model = gr.Textbox(
                        label="Base Model",
                        value=default_base_model,
                        placeholder="Base model name for PEFT"
                    )
                    
                    max_length = gr.Slider(
                        label="Max Length",
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10
                    )
                    
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1
                    )
                    
                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1
                    )
                    
                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=5
                    )
                    
                    num_sequences = gr.Slider(
                        label="Number of Responses",
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1
                    )
        
        # Display warning if model doesn't exist
        if not model_exists:
            gr.Markdown(
                """
                ⚠️ **Warning**: Model not found in the default location.
                
                Please:
                1. Run the data cleaning script: `python src/data_cleaning.py`
                2. Run the dataset preparation script: `python src/dataset_preparation.py`
                3. Run the model training script: `python src/model_training.py`
                
                Or update the model directory path in the Advanced Settings.
                """
            )
        
        # Set up event handler
        generate_button.click(
            fn=generate,
            inputs=[
                prompt,
                model_dir,
                base_model,
                max_length,
                temperature,
                top_p,
                top_k,
                num_sequences
            ],
            outputs=output
        )
        
        # Add examples
        with gr.Accordion("Example Prompts", open=True):
            example_prompts = [
                "The future of artificial intelligence is",
                "I always believed that the most important thing in life was",
                "If I could change one thing about our society, it would be",
                "The most beautiful thing I ever saw was"
            ]
            
            gr.Examples(
                examples=example_prompts,
                inputs=prompt
            )
        
        # Add instructions
        with gr.Accordion("Instructions", open=False):
            gr.Markdown(
                """
                ## How to Use
                
                1. Enter a prompt in the text box.
                2. Adjust the generation parameters if desired.
                3. Click "Generate" to create text in your writing style.
                
                ## Parameters
                
                - **Max Length**: Controls how long the generated text will be.
                - **Temperature**: Higher values (>1.0) make the output more random, while lower values make it more focused and deterministic.
                - **Top-p**: Controls diversity via nucleus sampling. Lower values will make the output more focused.
                - **Top-k**: Controls diversity by limiting to the k most likely next words. Lower values make the output more focused.
                - **Number of Responses**: Generate multiple different responses to the same prompt.
                """
            )
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_ui()
    interface.launch(share=False) 