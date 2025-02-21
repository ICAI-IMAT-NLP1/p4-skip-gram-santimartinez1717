from curses import window
from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file

    # Preprocess and tokenize the text

    tokens: List[str] = tokenize(text)


    return tokens

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """

    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.
    sorted_vocab: List[str] = list(sorted(word_counts, key=word_counts.get, reverse=True))
    

    
    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = {idx: word for idx, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: i for i, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the 
    presence of frequent words according to Mikolov's subsampling technique. This method 
    calculates the probability of keeping each word in the dataset based on its frequency, 
    with more frequent words having a higher chance of being discarded. The process helps 
    in balancing the word distribution, potentially leading to faster training and better 
    representations by focusing more on less frequent words.
    
    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.

        
    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """

    # Convert words to integers
    int_words: List[int] = [vocab_to_int[word] for word in words]


    word_counts: Counter = Counter(int_words)
    total_words = len(int_words)
    
    freqs: Dict[int, float] = {word: count / total_words for word, count in word_counts.items()}

    discard_probs: Dict[int, float] = {word: 1 - torch.sqrt(torch.tensor(threshold / freq)).item() for word, freq in freqs.items()}

    train_words: List[int] = [word for word in int_words if torch.rand(1).item() > discard_probs[word]]

    return train_words, freqs

def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:

    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """

    
    r = torch.randint(1, window_size + 1, (1,)).item()

    start = max(0, idx - r)
    end = min(len(words), idx + r + 1)

    history_words = words[start:idx]
    future_words = words[idx+1:end]
    target_words: List[str] = history_words + future_words

    return target_words

def get_batches(words: List[int], batch_size: int, window_size: int = 5) -> Generator[Tuple[List[int], List[int]],None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """
    inputs = []
    targets = []
    
    for idx in range(0, len(words)):

        target_word = words[idx]
        context = get_target(words, idx, window_size)

        for context_word in context:
            inputs.append(target_word)
            targets.append(context_word)

        if len(inputs) >= batch_size:
            yield inputs[:batch_size], targets[:batch_size]
            inputs = inputs[batch_size:]
            targets = targets[batch_size:]

def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    
    # Get the size of the vocabulary
    vocab_size = embedding.num_embeddings
    
    # Randomly select `valid_size` indices from the range of `valid_window`
    valid_examples = torch.randint(0, valid_window, (valid_size,), device=device)

    # Get the embeddings of the selected words
    valid_embeddings = embedding(valid_examples)  # (valid_size, embedding_dim)
    
    # Normalize the embeddings to calculate cosine similarity efficiently
    valid_norm = torch.nn.functional.normalize(valid_embeddings, p=2, dim=1)  # Normalize along the embedding dimension
    
    # Calculate the cosine similarities between the valid words and all the words in the vocabulary
    all_embeddings = embedding.weight  # All word embeddings (vocab_size, embedding_dim)
    all_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)  # Normalize the entire vocabulary
    
    # Cosine similarity is the dot product of normalized vectors
    similarities = torch.mm(valid_norm, all_norm.T)  # (valid_size, vocab_size)
    
    return valid_examples, similarities