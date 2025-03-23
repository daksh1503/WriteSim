"""
Text cleaning and preprocessing module for personal writings.
"""

import os
import re
import argparse
from glob import glob
from pathlib import Path
import nltk
nltk.download('punkt_tab')
import PyPDF2 

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def load_raw_texts(data_dir='data/raw'):
    """
    Load all .txt files from the specified directory.
    
    Args:
        data_dir (str): Directory containing raw text files
        
    Returns:
        dict: Dictionary mapping filenames to text content
    """
    raw_texts = {}
    
    # Process text files
    for filepath in glob(os.path.join(data_dir, '*.txt')):
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as file:
            raw_texts[filename] = file.read()
    
    # Process PDF files
    for filepath in glob(os.path.join(data_dir, '*.pdf')):
        filename = os.path.basename(filepath)
        extracted_text = extract_text_from_pdf(filepath)
        raw_texts[filename] = extracted_text
    
    print(f"Loaded {len(raw_texts)} text files from {data_dir}")
    return raw_texts

def extract_text_from_pdf(filepath):
    """
    Extract text from a PDF file.
    
    Args:
        filepath (str): Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    text = ""
    
    # Using PyPDF2
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
            
    return text

def clean_text(text):
    """
    Clean and normalize text.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    # Remove multiple line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common artifacts
    text = re.sub(r'--', '—', text)  # Convert double hyphens to em dashes
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Fix spacing after punctuation
    text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)
    
    # Restore paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def segment_into_chunks(text, max_length=512):
    """
    Segment text into reasonable chunks for model training.
    
    Args:
        text (str): Cleaned text
        max_length (int): Maximum chunk length
        
    Returns:
        list: List of text chunks
    """
    # Split text into sentences
    sentences = nltk.sent_tokenize(text, language='english')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Save current chunk and start a new one
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks


def process_texts(raw_texts, output_dir='data/processed'):
    """
    Process all texts and save them to the output directory.
    
    Args:
        raw_texts (dict): Dictionary of raw texts
        output_dir (str): Output directory
        
    Returns:
        list: List of all text chunks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_chunks = []
    
    for filename, text in raw_texts.items():
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Segment into chunks
        chunks = segment_into_chunks(cleaned_text)
        all_chunks.extend(chunks)
        
        # Save cleaned text
        clean_filename = os.path.join(output_dir, f"cleaned_{filename}")
        with open(clean_filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"Processed {filename}: {len(chunks)} chunks")
    
    # Save all chunks to a single file for model training
    with open(os.path.join(output_dir, 'all_chunks.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_chunks))
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description='Clean and process text files')
    parser.add_argument('--input', default='data/raw', help='Input directory with raw text files')
    parser.add_argument('--output', default='data/processed', help='Output directory for processed files')
    
    args = parser.parse_args()
    
    print(f"Loading texts from {args.input}...")
    raw_texts = load_raw_texts(args.input)
    
    if not raw_texts:
        print("No text files found. Please add .txt files to the data/raw directory.")
        return
    
    print(f"Processing texts and saving to {args.output}...")
    process_texts(raw_texts, args.output)
    print("Text processing complete!")


if __name__ == "__main__":
    main() 