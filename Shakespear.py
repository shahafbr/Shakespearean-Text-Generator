import nltk
import json
import random
import re
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams, ngrams
from collections import defaultdict
from nltk.corpus import gutenberg

# Download the Gutenberg dataset to load Shakespeare text:
nltk.download('gutenberg')

#---------------------------------------------- Task 1: Data Preparation ------------------------------------------#

# 1. Read Shakespeare's Works:
def preprocess_text(text):
    """
    Preprocess text:
      1. converting to lowercase.
       2. removing punctuation.
         3. splitting text into tokens.
    
    Inputs:
        text - Type:str | Data:Shakespeare input text to preprocess.
    Returns:
        tokens - Type:list | Data:List of Shakespeare text as cleaned tokens.
    """
    text = text.lower()  # Convert text to lowercase.
    text = re.sub(r'[^a-z\s]', '', text)  # Removing punctuation - only keeps letters and spaces.
    tokens = word_tokenize(text)  # Tokenizing the text - word based tokenization. 
    return tokens

def generate_bigrams(tokens):
    """
    Generating a list of bigrams from the preprocessed text
    
    Inputs:
        tokens - Type:list | Data:List of Shakespeare text as cleaned tokens.    
    Returns:
        bigram_l - Type:list | Data: A list of Tokenized Shakespeare text bigram tuples.
    """
    bigram_l = list(bigrams(tokens))  # Creating a bigram list using NLTK bigrams function.
    return bigram_l

# 2. Dictionary of Bigram Counts:
def generate_bigram_counts(tokens):
    """
    Generates a dictionary mapping bigrams to a dictionary of next-token counts.

    Inputs:
        tokens - Type:list | Data:List of tokens from preprocessed Shakespeare text.
    
    Returns:
        from_bigram_to_next_token_counts - Type:dict | Data: Dictionary where key is a bigram (tuple of two tokens) 
         and the value is a dictionary of counts of tokens that follow the bigram.
    """
    # Initializing defaultdict for automatic key creation:
    from_bigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(tokens) - 2):  # Making sure there's a next token.
        bigram = (tokens[i], tokens[i + 1])
        next_token = tokens[i + 2]
        from_bigram_to_next_token_counts[bigram][next_token] += 1  # Increment count.

    return dict(from_bigram_to_next_token_counts)

#------------------------------------------ Task 2: Probability Distribution --------------------------------------#

# Calculate bigram to next token Probabilities:
def compute_bigram_probabilities(from_bigram_to_next_token_counts):
    """
    Converts bigram to next token counts into probabilities.

    Inputs:
        from_bigram_to_next_token_counts - Type:dict | Data:A dictionary where keys are bigrams 
         and values are dictionaries containing the counts of next possible tokens.

    Returns:
        from_bigram_to_next_token_probs - Type:dict | Data:A dictionary where keys are bigrams and values are
        dictionaries of probabilities of next tokens.
    """
    from_bigram_to_next_token_probs = {}

    for bigram, next_token_counts in from_bigram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())  # Sum of all next word occurrences for computing probabilities.
        from_bigram_to_next_token_probs[bigram] = {
            token: count / total_count for token, count in next_token_counts.items()
        }

    return from_bigram_to_next_token_probs

#--------------------------------------------- Task 3: Sampling Next Token -----------------------------------------#

# Implement Sampling Function:
def sample_next_token(from_bigram_to_next_token_probs, bigram):
    """
    Samples the next token for a bigram based on the probability distribution.

    Inputs:
        from_bigram_to_next_token_probs - Type:dict | Data:A dictionary where keys are bigrams 
        and values are dictionaries mapping next tokens to their probabilities.
        bigram - Type:tuple(tuples of two tokens) | Data:The bigram for which we want to sample the next token from.

    Returns:
        next_token - Type:str | Data:A randomly sampled token based on the probability distribution.
    """

    # Check if there is a next token available for this input bigram:
    if bigram not in from_bigram_to_next_token_probs:
        return None  # There is no next token.

    next_tokens = list(from_bigram_to_next_token_probs[bigram].keys())
    probabilities = list(from_bigram_to_next_token_probs[bigram].values())

    # Sample a token using random weighted probabilities:
    next_token = random.choices(next_tokens, weights=probabilities, k=1)[0]
    return next_token

#----------------------------------------------- Task 4: Generating Text -------------------------------------------#

# Generate Text:
def generate_text_from_bigram(from_bigram_to_next_token_probs, initial_bigram, num_words):
    """
    Generates text by starting with an initial bigram and sampling the next token iteratively.

    Inputs:
        from_bigram_to_next_token_probs - Type:dict.
        initial_bigram - Type:tuple | Data:The starting bigram (tuple of two words).
        num_words - Type:int | Data: The number of next words we want to generate from our bigram.

    Returns:
        generated_text - Type:str | Data:The generated text from the input bigram.
    """
    if initial_bigram not in from_bigram_to_next_token_probs:
        return "Error: Initial bigram not found in probability dictionary."

    generated_tokens = list(initial_bigram)  # Start with the initial bigram

    for _ in range(num_words - 2):  # Already have two words, generate the rest
        current_bigram = tuple(generated_tokens[-2:])  # Last two words form the bigram
        next_token = sample_next_token(from_bigram_to_next_token_probs, current_bigram)

        if next_token is None:  # If no next token exists, stop generation
            break
        
        generated_tokens.append(next_token)  # Add the sampled token to the sequence

    return ' '.join(generated_tokens)  # Convert list of words into a string

#-------------------------------------- Task 5: Exploration of Different N-grams -----------------------------------#
# Experimenting with Trigrams and Quadgrams:
# Converting the function generate_bigrams to generate n-grams:
def generate_ngrams(tokens, n):
    """
    Generate a list of n-grams from a list of tokens.
    """
    return list(ngrams(tokens, n))

# Converting the function generate_bigram_counts to generate n-grams counts:
def generate_ngram_counts(tokens, n):
    """
    Generates a dictionary mapping n-grams to a dictionary of next-token counts.
    """
    from_ngram_to_next_token_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(tokens) - n):
        ngram = tuple(tokens[i:i+n])
        next_token = tokens[i + n]
        from_ngram_to_next_token_counts[ngram][next_token] += 1  # Increment count

    return dict(from_ngram_to_next_token_counts)

# Converting the function compute_bigram_probabilities to compute n-gram probabilities:
def compute_ngram_probabilities(from_ngram_to_next_token_counts):
    """
    Converts n-gram to next token counts into probabilities.
    """
    from_ngram_to_next_token_probs = {}

    for ngram, next_token_counts in from_ngram_to_next_token_counts.items():
        # Calculating the total amount of next tokens:
        total_count = sum(next_token_counts.values())
        # Convert from count to a probability:
        from_ngram_to_next_token_probs[ngram] = {
            token: count / total_count for token, count in next_token_counts.items()
        }

    return from_ngram_to_next_token_probs

# Converting the function generate_text_from_bigram to generate_text_from_ngram:
def generate_text_from_ngram(from_ngram_to_next_token_probs, initial_ngram, num_words):
    """
    Generates text by starting with an initial n-gram and sampling the next token iteratively.
    """
    if initial_ngram not in from_ngram_to_next_token_probs:
        return "Error: Initial n-gram not found in probability dictionary."

    generated_tokens = list(initial_ngram)

    for _ in range(num_words - len(initial_ngram)):
        current_ngram = tuple(generated_tokens[-len(initial_ngram):])  # Get the last n words.
        next_token = sample_next_token(from_ngram_to_next_token_probs, current_ngram)

        if next_token is None:  # If no next token exists, stop generation.
            break
        
        generated_tokens.append(next_token)

    return ' '.join(generated_tokens)


#---------------------------------------------- Task 6: Human Evaluation -------------------------------------------#
#  A Human Evaluation Survey:
def conduct_survey():
    """
    A function to survey users so I can evaluate the quality of my generated Shakespearean text.
    """
    print("\nPlease answer the following survey questions:")
    
    responses = {}
    responses["Name"] = input("Please enter your name: ")
    responses["familiarity"] = input("How familiar are you with Shakespearean language? (low/medium/high): ")
    responses["coherence"] = input("How coherent is the generated text? (1-5): ")
    responses["style"] = input("How well does it capture Shakespearean style? (1-5): ")
    responses["usefulness"] = input("Would you use this AI-generated text for creative writing, education, or entertainment? (yes/no): ")
    
    with open("survey_results.json", "a") as file:
        json.dump(responses, file)
        file.write("\n")
    
    print("Thank you! Your feedback has been recorded.")

def main():
    # Available Shakespeare texts in NLTK - ['shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt']
    # Loading Shakespeare text - 'shakespeare-hamlet.txt':
    shakespeare_text = gutenberg.raw('shakespeare-hamlet.txt')
    shakespeare_tokens = preprocess_text(shakespeare_text)
    
    # Testing each bigrams, trigrams, and quadgrams separetly based on the 
    # assignment specification. 

    # Generating bigrams, trigrams, and quadgrams:
    # bigrams_list = generate_bigrams(shakespeare_tokens)
    # trigrams_list = generate_ngrams(shakespeare_tokens, 3)
    # quadgrams_list = generate_ngrams(shakespeare_tokens, 4)

    # Generating the counts for each n-gram:
    # bigram_counts = generate_bigram_counts(shakespeare_tokens)
    # trigram_counts = generate_ngram_counts(shakespeare_tokens, 3)
    # quadgram_counts = generate_ngram_counts(shakespeare_tokens, 4)

    # Computing the probabilities for each n-gram model:
    # bigram_probs = compute_bigram_probabilities(bigram_counts)
    # trigram_probs = compute_ngram_probabilities(trigram_counts)
    # quadgram_probs = compute_ngram_probabilities(quadgram_counts)

    # Generating text using different n-grams:
    # bigram_text = generate_text_from_bigram(bigram_probs, ('to', 'be'), 10)
    # trigram_text = generate_text_from_ngram(trigram_probs, ('to', 'be', 'or'), 10)
    # quadgram_text = generate_text_from_ngram(quadgram_probs, ('to', 'be', 'or', 'not'), 10)

    # print("\nGenerated Text with Bigrams:")
    # print(bigram_text)
    # print("\nGenerated Text with Trigrams:")
    # print(trigram_text)
    # print("\nGenerated Text with Quadgrams:")
    # print(quadgram_text)

    while True:
        choice = input("Choose n-gram model (bigram/trigram/quadgram) or type 'exit' to quit: ").strip().lower()
        if choice == 'exit':
            break
        elif choice == 'bigram':
            n = 2
        elif choice == 'trigram':
            n = 3
        elif choice == 'quadgram':
            n = 4
        else:
            print("Invalid choice. Please enter 'bigram', 'trigram', or 'quadgram'.")
            continue

        # Generating the n-grams, there count and probability based on user input:
        ngram_list = generate_ngrams(shakespeare_tokens, n)
        ngram_counts = generate_ngram_counts(shakespeare_tokens, n)
        ngram_probs = compute_ngram_probabilities(ngram_counts)
        
        # Asking the user for the initial n-gram:
        user_input = input(f"Enter a custom {n}-gram (i.e., {n} words separated by spaces and only in lowercase !!!) or press enter to use default: ").strip()
        
        # Making sure user input maches there model selection:
        if user_input:
            custom_ngram = user_input.split()
            if len(custom_ngram) != n:
                print(f"Error: You must enter exactly {n} words. Using default n-gram instead.")
                initial_ngram = tuple(shakespeare_tokens[:n])
            else:
                initial_ngram = tuple(custom_ngram)
        else:
            # Use the default initial n-gram from the corpus if n-gram not specified: 
            initial_ngram = tuple(shakespeare_tokens[:n])
        
        # Generating text using the user inputed initial n-gram:
        generated_text = generate_text_from_ngram(ngram_probs, initial_ngram, 10)
        print("\nGenerated Text:")
        print(generated_text)
        
        # Adding a optional selection for the user to take a survey:
        next_step = input("Do you want to take the survey? (yes/no): ").strip().lower()
        if next_step == 'yes':
            conduct_survey()

if __name__ == "__main__":
    main()
