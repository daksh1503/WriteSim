# WriteSim: Fine-Tuning LLMs on Personal Writings

This project allows you to fine-tune a language model on your personal writings to generate text in your writing style.

## Features

- Data cleaning and preprocessing for text files
- Dataset preparation for transformer models
- Fine-tuning with efficient techniques (LoRA/PEFT)
- Text generation through command line or Gradio UI
- Metal acceleration support for M1/M2 Macs

## Setup

1. Place your text files in the `data/raw` (Create the directory manually after pulling the repo) directory
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the data cleaning script:
   ```
   python src/data_cleaning.py
   ```
4. Fine-tune the model:
   ```
   python src/model_training.py
   ```
5. Generate text with your model:
   ```
   python src/inference.py
   ```
   
Or use the Gradio UI:
```
python app.py
```

## Configuration

You can adjust training parameters in `src/model_training.py` including:
- Learning rate
- Number of epochs
- LoRA rank and alpha
- Model size
- Batch size 