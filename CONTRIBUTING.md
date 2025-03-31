# Contributing to WriteSim GPT-4

Thank you for your interest in contributing to WriteSim! This document provides guidelines for contributions.

## Getting Started

1. Fork the repository

2. Clone the Repo
   ```bash
   git clone https://github.com/your-username/WriteSim.git 
   cd WriteSim
   ``` 
3. Create a New Branch
   ```bash
   git checkout -b feature-branch
   ```
4. Make Your Changes
5. Commit Your Changes
   ```bash
   git add .
   git commit -m "Brief description of changes"
   git push origin feature-branch
   ```
6. Make your changes
7. Test locally:
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