"""
Gradio interface for WriteSim text generation using GPT-4 Turbo.
"""

import gradio as gr
import os
from openai import OpenAI, OpenAIError
from src.style_templates import STYLE_TEMPLATE
import sys

# Update the client initialization with error handling
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure OPENAI_API_KEY environment variable is set correctly")
    sys.exit(1)

def generate_text(
    prompt, 
    max_length=800,
    temperature=0.7,
    top_p=0.9,
    style_intensity="moderate"
):
    """Generate text using GPT-4 Turbo with the specified style."""
    try:
        # Get style template
        style = STYLE_TEMPLATE["introspective"]
        
        # Adjust system message based on style intensity
        intensity_prefix = {
            "subtle": "While maintaining readability and natural flow, occasionally incorporate",
            "moderate": "Consistently use",
            "strong": "Strictly adhere to"
        }
        
        system_message = f"{intensity_prefix[style_intensity]} {style['system_message']}"
        
        # Create few-shot examples
        examples_prompt = "Here are examples of the style:\n\n" + "\n".join(
            f"{i+1}. {example}" for i, example in enumerate(style['examples'])
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": examples_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
        except OpenAIError as e:
            return f"OpenAI API error: {str(e)}"
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating text: {e}"

# Create the Gradio interface
with gr.Blocks(title="WriteSim GPT-4", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Advanced AI made easy
    Overcome writer's block with our AI writing assistant.
    """, elem_classes=["center-text"])
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            # Main content area
            prompt = gr.Textbox(
                label="",
                placeholder="Enter your prompt here...",
                lines=5,
                elem_classes=["clean-textbox"]
            )
            
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="lg")
            
            output = gr.Textbox(
                label="",
                lines=8,
                show_copy_button=True,
                elem_classes=["clean-textbox"]
            )
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Generation Settings")
                max_length = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=800,
                    step=100,
                    label="Maximum Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                style_intensity = gr.Radio(
                    choices=["subtle", "moderate", "strong"],
                    value="moderate",
                    label="Style Intensity"
                )
    
    # Add footer
    gr.Markdown("""
    ---
    © 2025 WriteSim. All rights reserved.
    """, elem_classes=["footer"])

    # Add custom CSS
    css = """
    .center-text {
        text-align: center;
        margin-bottom: 2rem;
    }
    .center-text h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .clean-textbox {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: white;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        position: fixed;
        bottom: 0;
        width: 100%;
        background: white;
        border-top: 1px solid #e5e7eb;
    }
    """
    demo.load(None, None, None, _js=f"() => {{ document.head.innerHTML += `<style>{css}</style>` }}")

    # Set up event handlers
    generate_btn.click(
        generate_text,
        inputs=[prompt, max_length, temperature, style_intensity],
        outputs=output
    )
    clear_btn.click(lambda: "", None, prompt)
    clear_btn.click(lambda: "", None, output)

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    # Launch with custom theme
    demo.launch(
        share=False,
        theme=gr.themes.Soft(
            primary_hue="slate",
            neutral_hue="slate",
            font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
        )
    )