# WriteSim: Advanced AI Writing Assistant

A minimalist GPT-4 powered writing assistant that helps you generate text in a specific style.

## Overview

WriteSim uses GPT-4 Turbo to generate high-quality text that matches a predefined writing style. Perfect for writers, content creators, and anyone looking to maintain consistent writing across their work.

## Features

- Clean, minimalist interface
- GPT-4 Turbo powered text generation
- Customizable style intensity (subtle, moderate, strong)
- Adjustable generation parameters
- One-click text generation

## Quick Start

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate     # On Windows
   
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

## Usage

1. Enter your writing prompt in the text box
2. Adjust generation settings if needed:
   - Maximum Length: Control the length of generated text
   - Temperature: Adjust creativity level
   - Style Intensity: Choose between subtle, moderate, or strong style adherence
3. Click "Generate" to create your text
4. Use the copy button to copy the generated text

## Requirements

- Python 3.8+
- OpenAI API key
- Minimal dependencies:
  - openai
  - gradio
  - typing-extensions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

© 2025 WriteSim. All rights reserved. 