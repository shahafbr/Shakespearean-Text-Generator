# Shakespearean Text Generator:

## Overview
This project implements an N-gram-based text generator that mimics the writing style of William Shakespeare. The generator can create text using bigrams, trigrams, and quadgrams, leveraging probability distributions to determine the next token in a sequence.

## Features
- Preprocesses Shakespeare's works by tokenizing text and removing punctuation.
- Supports text generation using bigrams, trigrams, and quadgrams.
- Implements probability distributions for accurate word prediction.
- Allows for weighted random sampling of words.
- Generates text based on an initial n-gram.
- Includes a human evaluation component for assessing output quality.

## Installation instructions
1. Ensure You Have Python Installed:
```bash
    python --version
```
2. Install nltk:
```bash
   pip install nltk
```
3. Download the Gutenberg corpus and necessary tokenizers (like 'punkt'):
```bash
   python -m nltk.downloader gutenberg
   python -m nltk.downloader punkt
```
or run the Download.py

## File Structure
```plaintext
Shakespeare_Text_Generator/
│── Shakespeare.md
│── Download.py            # Installation
│── Test.py                # Testing script for validating Shakespeare.py functionality. 
│── Ssurvey_results.json   # File to store user servey responses for evaluation. 
│── Shakespeare.py         # Main script to generate text.
```
## Overall Approach
1. Preparation: The code starts by loading and preprocessing Shakespeare’s text (from nltk - gutenberg).
2. Model Building: Then we constructs n-grams (starting with bigrams) and computing the corresponding counts and probability distributions.
3. Sampling: We use a weighted random choice to sample the next token to the input n-gram, based on the probability distributions we computed previously.
4. Text Generation: We generate each word by iteratively sampling tokens based on the input n-gram. Using this approach we generate a sequences of tokens to create our sentance.
5. Exploration: The approach we used is extended to different n-gram sizes (trigrams, quadgrams) to experiment with text generation using different contexts.
6. Testing: We use a testing python script to test our models using 10 different test cases which are explained below. 
7. Evaluation: Finally, we have a human feedback survey, which is gathered in a jason file to assess the quality of the generated text.

## Testing 
```plaintext
Shakespeare_Text_Generator/
│── Test 1 - Preprocess Text
│── Test 2 - Generate Bigrams
│── Test 3 - Generate Bigram Counts
│── Test 4 - Compute Bigram Probabilitie
│── Test 5 - Sample Next Token
│── Test 6 - Generate Text From Bigram
│── Test 7 - Generate n-grams
│── Test 8 - Generate n-gram Counts
│── Test 9 - Compute n-gram Probabilities
│── Test 10 - Generate Text From n-gram
```

