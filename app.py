"""
Gradio interface for WriteSim text generation using GPT-4 Turbo.
"""

import gradio as gr
import os
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_writing_style():
    """Load the analyzed writing style patterns."""
    try:
        with open('data/processed/style_patterns.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def generate_text(
    prompt, 
    max_length=800,
    temperature=0.7,
    top_p=0.9,
    style_intensity="moderate"
):
    """Generate text using GPT-4 Turbo with your writing style."""
    try:
        style_patterns = load_writing_style()
        if not style_patterns:
            return "Error: Writing style patterns not found. Please analyze your writing first."

        # Create a system message that instructs GPT-4 to mimic your style
        system_message = f"""You are a writing assistant that mimics the following style characteristics:
- Typical sentence length: {style_patterns['avg_sentence_length']} words
- Common phrase patterns: {', '.join(style_patterns['common_phrases'][:5])}
- Vocabulary level: {style_patterns['vocabulary_level']}
- Tone: {style_patterns['tone']}
- Writing quirks: {', '.join(style_patterns['writing_quirks'])}

Maintain these style elements while generating text that sounds natural and coherent."""

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # GPT-4 Turbo
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating text: {e}"

# Create the Gradio interface
with gr.Blocks(title="WriteSim GPT-4", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # WriteSim: Your Personal Writing Style Generator
    Powered by GPT-4 Turbo
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
                    minimum=100,
                    maximum=2000,
                    value=800,
                    step=100,
                    label="Maximum Length (tokens)"
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
                style_intensity = gr.Radio(
                    choices=["subtle", "moderate", "strong"],
                    value="moderate",
                    label="Style Intensity"
                )
    
    # Set up event handlers
    generate_btn.click(
        generate_text,
        inputs=[prompt, max_length, temperature, top_p, style_intensity],
        outputs=output
    )
    clear_btn.click(lambda: "", None, prompt)
    clear_btn.click(lambda: "", None, output)

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    demo.launch(share=False) 