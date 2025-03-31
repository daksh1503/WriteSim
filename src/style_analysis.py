"""
Analyze writing style from raw text files.
"""

import os
import nltk
import json
from collections import Counter
from textstat import textstat

def analyze_writing_style(data_dir='data/raw'):
    """Analyze writing style patterns from raw text files."""
    
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    all_text = ""
    for filepath in os.listdir(data_dir):
        if filepath.endswith(('.txt', '.pdf')):
            with open(os.path.join(data_dir, filepath), 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n\n"
    
    # Basic text analysis
    sentences = nltk.sent_tokenize(all_text)
    words = nltk.word_tokenize(all_text)
    
    # Analyze sentence length
    sentence_lengths = [len(nltk.word_tokenize(sent)) for sent in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # Analyze vocabulary using NLTK
    pos_tags = nltk.pos_tag(words)
    word_types = Counter(tag for word, tag in pos_tags)
    common_words = Counter(words).most_common(20)
    
    # Analyze writing level and style
    writing_level = textstat.flesch_kincaid_grade(all_text)
    
    # Detect writing quirks
    quirks = []
    if len([w for w in words if w.isupper()]) / len(words) > 0.05:
        quirks.append("frequent capitalization")
    if all_text.count("...") > len(sentences) * 0.1:
        quirks.append("uses ellipsis frequently")
    if all_text.count("!") > len(sentences) * 0.2:
        quirks.append("emphatic exclamations")
    if word_types.get('JJ', 0) / len(words) > 0.1:
        quirks.append("frequent use of adjectives")
    
    # Save analysis results
    style_patterns = {
        "avg_sentence_length": round(avg_sentence_length, 1),
        "common_words": [word for word, count in common_words],
        "vocabulary_level": writing_level,
        "tone": "formal" if writing_level > 10 else "casual",
        "writing_quirks": quirks,
        "sentence_variety": {
            "short": len([l for l in sentence_lengths if l < 10]),
            "medium": len([l for l in sentence_lengths if 10 <= l <= 20]),
            "long": len([l for l in sentence_lengths if l > 20])
        }
    }
    
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/style_patterns.json', 'w') as f:
        json.dump(style_patterns, f, indent=2)
    
    print("Writing style analysis complete!")
    return style_patterns

if __name__ == "__main__":
    analyze_writing_style() 