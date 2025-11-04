import re
import transformers

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import GPT2Tokenizer

# ==========================================================
#  Character-level Tokenizer
# ==========================================================
class CharTokenizer:
    """
    Character-level tokenizer that encodes text as individual characters
    (plus a few special tokens for structure).
    
    Suitable for small corpora or stylistically constrained tasks.
    """
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.special_tokens = ['<TITLE>', '</TITLE>', '<NL>', '<STANZA>', '<POEM>', '<UNK>']

    def fit(self, text):
        """
        Build a vocabulary directly from a raw text string.
        
        Args:
            text (str): training text to extract characters from.
        """
        idx = 0
        # Add special tokens first
        for tok in self.special_tokens:
            self.vocab[tok] = idx
            idx += 1
        
        # Add all unique characters from text
        for char in text:
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1
        
        # Build reverse mapping
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def encode(self, text):
        """
        Encode text into integer token IDs, keeping special tokens intact.
        """
        tokens = re.findall(r'<[^>]+>|.', text, flags=re.UNICODE)
        return [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]

    def decode(self, indices):
        """
        Decode a list of token IDs back to readable text.
        """
        tokens = [self.inv_vocab.get(i, '<UNK>') for i in indices]

        # Map special tokens to readable output
        special_map = {
            '<TITLE>': '',
            '</TITLE>': '',
            '<NL>': '\n',
            '<STANZA>': '\n\n',
            '<POEM>': '\n\n\n\n\n',
        }
        tokens = [special_map.get(t, t) for t in tokens]

        return ''.join(tokens)


# ==========================================================
#  Byte-Pair Encoding (BPE) Tokenizer
# ==========================================================
class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer using Hugging Face's Tokenizers library.
    
    Trains subword units on text files and supports custom structural tokens.
    """
    def __init__(self, vocab_size=5000):
        """
        Args:
            vocab_size (int): maximum vocabulary size for BPE training.
        """
        self.vocab_size = vocab_size
        self.special_tokens = ['<TITLE>', '</TITLE>', '<NL>', '<STANZA>', '<POEM>', '<UNK>']

        # Initialize BPE model and trainer
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )

    def fit(self, files):
        """
        Train the BPE tokenizer on a list of text files.
        
        Args:
            files (List[str]): list of text file paths.
        """
        self.tokenizer.train(files, self.trainer)

    def encode(self, text):
        """
        Encode text into subword token IDs.
        """
        return self.tokenizer.encode(text).ids

    def decode(self, indices):
        """
        Decode a list of token IDs into readable text, handling structural markers.
        """
        tokens = [self.tokenizer.id_to_token(i) for i in indices]

        # Map special tokens to readable structure
        special_map = {
            '<TITLE>': '',
            '</TITLE>': '',
            '<NL>': '\n',
            '<STANZA>': '\n\n',
            '<POEM>': '\n\n\n\n\n',
        }
        tokens = [special_map.get(t, t) for t in tokens]

        text = ''.join(tokens)
        text = text.replace('Ġ', ' ')   # Fix ByteLevel space markers
        text = text.replace('âĢĶ', '-') # Handle stray encoding artifacts
        return text


# ==========================================================
#  GPT-2 Tokenizer Wrapper
# ==========================================================
class GPT2TokenizerWrapper:
    """
    Wrapper around Hugging Face GPT-2 tokenizer for consistency with CharTokenizer and BPETokenizer.
    Adds custom structural tokens and exposes a unified encode/decode interface.
    """
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.special_tokens = ['<TITLE>', '</TITLE>', '<NL>', '<STANZA>', '<POEM>']

        # Add custom tokens to GPT-2's vocabulary
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        self.vocab_size = len(self.tokenizer)

    def encode(self, text):
        """
        Encode text into GPT-2 token IDs.
        Keeps special tokens as single units.
        """
        transformers.logging.set_verbosity_error()  # silence tokenizer warnings
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, indices):
        """
        Decode a list of GPT-2 token IDs into readable text.
        Structural tokens are converted into newlines or spacing.
        """
        text = self.tokenizer.decode(indices, skip_special_tokens=False)
        
        special_map = {
            '<TITLE>': '',
            '</TITLE>': '',
            '<NL>': '\n',
            '<STANZA>': '\n\n',
            '<POEM>': '\n\n\n\n\n'
        }

        for tok, repl in special_map.items():
            text = text.replace(tok, repl)

        return text