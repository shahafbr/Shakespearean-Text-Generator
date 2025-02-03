import unittest
import random

# Import the functions from Shakespear.py:
from Shakespear import (
    preprocess_text,
    generate_bigrams,
    generate_bigram_counts,
    compute_bigram_probabilities,
    sample_next_token,
    generate_text_from_bigram,
    generate_ngrams,
    generate_ngram_counts,
    compute_ngram_probabilities,
    generate_text_from_ngram
)

# Testing Class for Shakespear.py functions: 
class TestShakespearFunctions(unittest.TestCase):
    def setUp(self):
        # A sample text for testing:
        self.sample_text = "To be, or not to be, that is the question."
        self.tokens = preprocess_text(self.sample_text)
    
    # Test 1 - Preprocess Text:
    # Verifying that preprocess_text converts text to lowercase, removes punctuation, and tokenizes correctly.
    def test_preprocess_text(self):
        expected_tokens = ['to', 'be', 'or', 'not', 'to', 'be', 'that', 'is', 'the', 'question']
        self.assertEqual(self.tokens, expected_tokens,
                         "preprocess_text did not return the expected tokens.")
        print("Test 1: successful")
    
    # Test 2 - Generate Bigrams:
    # Verifying that generate_bigrams produces the correct list of bigram tuples.
    def test_generate_bigrams(self):
        bigrams_list = generate_bigrams(self.tokens)
        expected_bigrams = [
            ('to', 'be'),
            ('be', 'or'),
            ('or', 'not'),
            ('not', 'to'),
            ('to', 'be'),
            ('be', 'that'),
            ('that', 'is'),
            ('is', 'the'),
            ('the', 'question')
        ]
        self.assertEqual(bigrams_list, expected_bigrams,
                         "generate_bigrams did not produce the expected bigrams.")
        print("Test 2: successful")
    
    # Test 3 - Generate Bigram Counts:
    # Verifying that generate_bigram_counts correctly counts the occurrences of next tokens for each bigram.
    def test_generate_bigram_counts(self):
        counts = generate_bigram_counts(self.tokens)
        # Testing for the bigram ('to', 'be'), where we expect two different next tokens: 'or' and 'that':
        self.assertIn(('to', 'be'), counts, "Bigram ('to', 'be') not found in counts.")
        self.assertEqual(counts[('to', 'be')]['or'], 1,
                         "Expected count for ('to', 'be') -> 'or' is not correct.")
        self.assertEqual(counts[('to', 'be')]['that'], 1,
                         "Expected count for ('to', 'be') -> 'that' is not correct.")
        
        # Testing for ('be', 'or'), where the next token should be 'not':
        self.assertEqual(counts[('be', 'or')]['not'], 1,
                         "Expected count for ('be', 'or') -> 'not' is not correct.")
        print("Test 3: successful")
    
    # Test 4 - Compute Bigram Probabilities:
    # Verifying that compute_bigram_probabilities calculates probabilities that sum to 1 for each bigram.
    def test_compute_bigram_probabilities(self):
        counts = generate_bigram_counts(self.tokens)
        probs = compute_bigram_probabilities(counts)
        for token_dict in probs.values():
            self.assertAlmostEqual(sum(token_dict.values()), 1.0,
                                   msg="Probabilities for a bigram do not sum to 1.")
        print("Test 4: successful")
    
    # Test 5 - Sample Next Token:
    # Verifying that sample_next_token returns a valid token based on the probability distribution,
    # and returns None when the input n-gram is not present in our curpus.
    def test_sample_next_token(self):
        fake_distribution = {('a', 'b'): {'c': 0.5, 'd': 0.5}}
        sampled = sample_next_token(fake_distribution, ('a', 'b'))
        self.assertIn(sampled, ['c', 'd'],
                      "sample_next_token did not return an expected token.")
        self.assertIsNone(sample_next_token(fake_distribution, ('x', 'y')),
                          "sample_next_token should return None for a missing bigram.")
        print("Test 5: successful")
    
    # Test 6 - Generate Text From Bigram:
    # Verifying that generate_text_from_bigram produces text with the correct number of words and
    # that it starts with the provided inputed bigram.
    def test_generate_text_from_bigram(self):
        counts = generate_bigram_counts(self.tokens)
        probs = compute_bigram_probabilities(counts)
        generated_text = generate_text_from_bigram(probs, ('to', 'be'), 6)
        generated_tokens = generated_text.split()
        self.assertEqual(len(generated_tokens), 6,
                         "Generated text does not have the expected number of words.")
        self.assertEqual(generated_tokens[:2], ['to', 'be'],
                         "Generated text does not start with the expected bigram.")
        print("Test 6: successful")
    
    # Test 7 - Generate n-grams:
    # Verifying that generate_ngrams correctly produces a list of trigrams (3-grams) from the token list.
    def test_generate_ngrams(self):
        trigrams = generate_ngrams(self.tokens, 3)
        expected_trigrams = [
            ('to', 'be', 'or'),
            ('be', 'or', 'not'),
            ('or', 'not', 'to'),
            ('not', 'to', 'be'),
            ('to', 'be', 'that'),
            ('be', 'that', 'is'),
            ('that', 'is', 'the'),
            ('is', 'the', 'question')
        ]
        self.assertEqual(trigrams, expected_trigrams,
                         "generate_ngrams did not return the expected trigrams.")
        print("Test 7: successful")
    
    # Test 8 - Generate n-gram Counts:
    # Verifying that generate_ngram_counts correctly counts the next token for a given trigram.
    def test_generate_ngram_counts(self):
        trigram_counts = generate_ngram_counts(self.tokens, 3)
        self.assertEqual(trigram_counts[('to', 'be', 'or')]['not'], 1,
                         "generate_ngram_counts did not count the trigram correctly.")
        print("Test 8: successful")
    
    # Test 9 - Compute n-gram Probabilities:
    # Verifying that compute_ngram_probabilities calculates probabilities that sum up to 1 for each trigram.
    def test_compute_ngram_probabilities(self):
        trigram_counts = generate_ngram_counts(self.tokens, 3)
        trigram_probs = compute_ngram_probabilities(trigram_counts)
        for token_dict in trigram_probs.values():
            self.assertAlmostEqual(sum(token_dict.values()), 1.0,
                                   msg="Probabilities for a trigram do not sum to 1.")
        print("Test 9: successful")
    
    # Test 10 - Generate Text From n-gram:
    # Verifying that generate_text_from_ngram produces text with the correct number of words and
    # that it starts with the inputed initial trigram.
    def test_generate_text_from_ngram(self):
        trigram_counts = generate_ngram_counts(self.tokens, 3)
        trigram_probs = compute_ngram_probabilities(trigram_counts)
        generated_text = generate_text_from_ngram(trigram_probs, ('to', 'be', 'or'), 7)
        generated_tokens = generated_text.split()
        self.assertEqual(len(generated_tokens), 7,
                         "Generated text from n-gram does not have the expected number of words.")
        self.assertEqual(generated_tokens[:3], ['to', 'be', 'or'],
                         "Generated text does not start with the expected trigram.")
        print("Test 10: successful")

if __name__ == '__main__':
    # Setting a random seed for consistency in our tests:
    random.seed(42)
    unittest.main()
