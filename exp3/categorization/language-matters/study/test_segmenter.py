#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reasoning Step Segmenter

This script takes a paragraph of reasoning text and segments it into distinct reasoning steps.
It uses a pretrained classifier to determine when one step ends and another begins.

Usage:
    python step_segmenter.py --text "Your reasoning paragraph here."
    python step_segmenter.py --file input.txt
"""

import argparse
import torch
import nltk
import os
import re
from sentence_transformers import SentenceTransformer
from torch import nn
import json

nltk.download('punkt_tab')
# Ensure nltk punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define the classifier model (must match the trained model architecture)
class SentencePairClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512):
        """
        Classifier for sentence pair embeddings that determines if they're different steps
        
        Args:
            embedding_dim (int): Dimension of sentence embeddings
            hidden_dim (int): Dimension of hidden layer
        """
        super(SentencePairClassifier, self).__init__()
        # Input: concatenated embeddings and cosine similarity
        self.fc1 = nn.Linear(embedding_dim * 2 + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, embeddings1, embeddings2):
        """
        Forward pass
        
        Args:
            embeddings1 (torch.Tensor): Embeddings of first sentences
            embeddings2 (torch.Tensor): Embeddings of second sentences
        
        Returns:
            torch.Tensor: Raw logits (apply sigmoid for probabilities)
        """
        # Compute cosine similarity between embeddings
        cosine_sim = torch.sum(embeddings1 * embeddings2, dim=1, keepdim=True)
        
        # Concatenate embeddings and similarity
        combined = torch.cat((embeddings1, embeddings2, cosine_sim), dim=1)
        
        # Pass through layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_models(model_path='best_f1_sentence_pair_classifier.pt', embedding_model_name='BAAI/bge-base-en-v1.5'):
    """
    Load the embedding model and classifier
    
    Args:
        model_path (str): Path to the trained classifier model
        embedding_model_name (str): Name of the sentence transformer model
        
    Returns:
        tuple: (classifier, sentence_transformer)
    """
    print(f"Loading embedding model: {embedding_model_name}")
    sentence_model = SentenceTransformer(embedding_model_name)
    
    # Get embedding dimension
    sample_embedding = sentence_model.encode("Sample text", convert_to_tensor=True)
    embedding_dim = sample_embedding.size(0)
    
    print(f"Loading classifier from: {model_path}")
    classifier = SentencePairClassifier(embedding_dim)
    
    # Check if model exists, if not, give instructions
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found.")
        print("Please ensure you've trained the model first or provide the correct path.")
        print("You can train the model using the training script provided earlier.")
        exit(1)
    
    # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    return classifier, sentence_model, device


def split_into_sentences(text):
    """
    Split text into sentences using NLTK sentence tokenizer
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sentences
    """
    # Clean text - remove excess whitespace and normalize
    # text = re.sub(r'\s+', ' ', text).strip()
    
    # # Use NLTK to split into sentences
    # sentences = nltk.sent_tokenize(text)
    sentences = text.split('\n')
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    # print(sentences)
    return sentences


def predict_pair(classifier, sentence_model, sentence1, sentence2, device):
    """
    Predict if two sentences belong to different steps
    
    Args:
        classifier (SentencePairClassifier): Trained classifier
        sentence_model (SentenceTransformer): Sentence transformer model
        sentence1 (str): First sentence
        sentence2 (str): Second sentence
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        tuple: (probability, predicted_label)
    """
    # Encode sentences
    embedding1 = sentence_model.encode(sentence1, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0).to(device)
    embedding2 = sentence_model.encode(sentence2, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = classifier(embedding1, embedding2)
        prob = torch.sigmoid(output).item()
    
    # Return probability and predicted label
    label = 1 if prob > 0.7 else 0

    return prob, label


def segment_text(text, classifier, sentence_model, device, threshold=0.5, verbose=False):
    """
    Segment text into reasoning steps
    
    Args:
        text (str): Input text
        classifier (SentencePairClassifier): Trained classifier
        sentence_model (SentenceTransformer): Sentence transformer model
        device (str): Device to run on
        threshold (float): Threshold for considering sentences as different steps
        verbose (bool): Whether to print detailed information
        
    Returns:
        list: List of steps, where each step is a list of sentences
    """
    # Split text into sentences
    sentences = split_into_sentences(text)
    
    if len(sentences) <= 1:
        return [sentences]
    
    # Initialize steps with the first sentence
    steps = [[sentences[0]]]
    
    # Process each consecutive pair of sentences
    for i in range(1, len(sentences)):
        current_sentence = sentences[i].strip()
        previous_sentence = sentences[i-1].strip()
        
        # Predict if this is a new step
        prob, is_new_step = predict_pair(classifier, sentence_model, previous_sentence, current_sentence, device)
        if current_sentence[:4].lower() == 'wait':
            is_new_step = True
        
        if verbose:
            print(f"\nSentence {i}:")
            print(f"Previous: '{previous_sentence}'")
            print(f"Current: '{current_sentence}'")
            print(f"Probability of new step: {prob:.4f} (Label: {is_new_step})")
        
        if is_new_step:
            # Start a new step
            steps.append([current_sentence])
        else:
            # Continue the current step
            steps[-1].append(current_sentence)
    
    return steps


def format_steps(steps):
    """
    Format steps for output
    
    Args:
        steps (list): List of steps, where each step is a list of sentences
        
    Returns:
        str: Formatted output
    """
    result = []
    for i, step in enumerate(steps, 1):
        step_text = " ".join(step)
        result.append(f"Step {i}: {step_text}")
    
    return "\n\n".join(result)


def format_steps_json(steps):
    """
    Format steps as JSON
    
    Args:
        steps (list): List of steps, where each step is a list of sentences
        
    Returns:
        str: JSON formatted output
    """
    result = []
    for i, step in enumerate(steps, 1):
        step_text = " ".join(step)
        result.append({
            "step_number": i,
            "text": step_text,
            "sentences": step
        })
    
    return json.dumps({"steps": result}, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Segment reasoning text into distinct steps.')
    
    # Input arguments (either text or file)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Input text to segment')
    input_group.add_argument('--file', type=str, help='Path to file containing text to segment')
    
    # Optional arguments
    parser.add_argument('--model', type=str, default='best_f1_sentence_pair_classifier.pt',
                      help='Path to the trained model file')
    parser.add_argument('--embedding-model', type=str, default='intfloat/multilingual-e5-base',
                      help='Name of the sentence transformer model')
    parser.add_argument('--output', type=str, help='Output file to write results')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    # Get input text
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        text = args.text
    
    # Load models
    classifier, sentence_model, device = load_models(args.model, args.embedding_model)
    
    # Segment text
    steps = segment_text(text, classifier, sentence_model, device, verbose=args.verbose)
    
    # Format output
    if args.json:
        output = format_steps_json(steps)
    else:
        output = format_steps(steps)
    
    # Write or print output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print("\nSegmented Steps:")
        print(output)


if __name__ == "__main__":
    main()