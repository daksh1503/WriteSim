"""
Gradio interface for WriteSim text generation.
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_model(model_path="models/fine_tuned"):
    """Load the fine-tuned model and tokenizer."""
    try:
        base_model = "gpt2"  # or load from config
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def generate_text(
    prompt, 
    max_length=200, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=50, 
    num_return_sequences=1,
    repetition_penalty=1.1
):
    """Generate text based on the prompt."""
    try:
        model, tokenizer = load_model()
        if model is None:
            return "Error: Model not found. Please train the model first."

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
        
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts[0]
    except Exception as e:
        return f"Error generating text: {e}"

def check_model_status():
    """Check if the model has been trained."""
    if os.path.exists("models/fine_tuned"):
        return "✅ Model is ready"
    return "❌ Model not found. Please train the model first."

# Create the Gradio interface
with gr.Blocks(title="WriteSim", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # WriteSim: Your Personal Writing Style Generator
    Generate text that matches your writing style using a fine-tuned language model.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            )
            
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear")
            
            output = gr.Textbox(
                label="Generated Text",
                lines=10,
                show_copy_button=True
            )
            
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Generation Settings")
                max_length = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=200,
                    step=50,
                    label="Maximum Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Top-p"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k"
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.1,
                    label="Repetition Penalty"
                )
                num_return_sequences = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Responses"
                )
            
            with gr.Group():
                gr.Markdown("### Model Status")
                status = gr.Markdown(check_model_status())
    
    with gr.Accordion("Help", open=False):
        gr.Markdown("""
        ### How to use WriteSim
        1. Enter a prompt in the text box
        2. Adjust the generation settings if desired:
           - **Maximum Length**: Controls how long the generated text will be
           - **Temperature**: Higher values make the output more creative but less focused
           - **Top-p**: Controls diversity of word choices
           - **Top-k**: Limits the number of tokens considered for each step
           - **Repetition Penalty**: Helps prevent repetitive text
           - **Number of Responses**: How many different completions to generate
        3. Click "Generate" to create text in your style
        
        ### Tips
        - Use longer prompts for more context
        - Adjust temperature up for creative writing, down for more focused output
        - If the output is too repetitive, increase the repetition penalty
        - For more variety, increase both top-k and top-p values
        """)
    
    # Set up event handlers
    generate_btn.click(
        generate_text,
        inputs=[prompt, max_length, temperature, top_p, top_k, num_return_sequences, repetition_penalty],
        outputs=output
    )
    clear_btn.click(lambda: "", None, prompt)
    clear_btn.click(lambda: "", None, output)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False) 