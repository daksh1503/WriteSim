# Contributing to WriteSim GPT-4

Thank you for your interest in contributing to WriteSim! This document provides guidelines for contributions.

## Getting Started

1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'  # Unix/macOS
   # or
   set OPENAI_API_KEY=your-api-key-here       # Windows
   ```
5. Make your changes
6. Test locally:
   ```bash
   python app.py
   ```
7. Submit a pull request

## Code Style

- Add docstrings for new functions
- Include type hints where appropriate
- Use f-strings for string formatting
- Keep functions focused and single-purpose

## Testing

Before submitting a PR, please:
1. Test style analysis with different text samples
2. Verify GPT-4 integration works
3. Check UI responsiveness
4. Ensure error messages are helpful 