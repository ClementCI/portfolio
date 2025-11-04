import torch
import random
import numpy as np
import os
import re
import csv

from torch.utils.data import TensorDataset, DataLoader
from .tokenizers import CharTokenizer, BPETokenizer, GPT2TokenizerWrapper

# ==========================================================
#  Helper Functions
# ==========================================================
def set_seed(seed):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_config(config):
    """
    Extract common training configuration parameters from a config dict.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple of parameters for training function.
    """
    return (
        config['model'],
        config['lr'],
        config['n_epochs'],
        config['batch_size'],
        config['block_size'],
        config['print_every'],
        config['eval_every'],
        config['n_generate'],
        config['tok_type']
    )


def create_loader(tensor_data, batch_size, shuffle):
    """
    Create a PyTorch DataLoader from a tensor dataset.

    Args:
        tensor_data (torch.Tensor): Input tensor.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle dataset.

    Returns:
        DataLoader object.
    """
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_tokenizer(book_fname='data/dickinson.txt', tok_type="char"):
    """
    Initialize and return a tokenizer of the specified type.

    Args:
        book_fname (str): Path to text file for tokenizer training (if needed).
        tok_type (str): 'char', 'bpe', or 'gpt2'.

    Returns:
        tokenizer object
    """
    if tok_type == "char":
        tokenizer = CharTokenizer()
        with open(book_fname, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer.fit(text)
    elif tok_type == "bpe":
        tokenizer = BPETokenizer()
        tokenizer.fit([book_fname])
    elif tok_type == "gpt2":
        tokenizer = GPT2TokenizerWrapper()
    else:
        raise ValueError("tok_type must be 'char', 'bpe', or 'gpt2'")

    return tokenizer


def get_datasets(book_fname='data/dickinson.txt',
                 device='cpu',
                 tokenizer=None,
                 block_size= 128,
                 config= None,
                 val_frac= 0.2,
                 test_frac= 0.1):
    """
    Prepare and encode datasets for training, validation, and testing.
    Applies overlapping chunks for GPT/GPT2 models.

    Args:
        book_fname (str): Path to raw text file.
        device (str): 'cpu' or 'cuda'.
        tokenizer: Fitted tokenizer (CharTokenizer, BPETokenizer, GPT2TokenizerWrapper).
        block_size (int): Sequence length per training example.
        config (dict): Model configuration containing 'type'.
        val_frac (float): Fraction for validation split.
        test_frac (float): Fraction for test split.

    Returns:
        dict of torch.Tensor blocks: {'train', 'val', 'test'}
    """
    # ------------------ Read and clean raw text ------------------
    with open(book_fname, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()

        # Pure Roman numeral lines -> treat as poem start
        is_roman = re.fullmatch(r'[IVXLCDM]+\.?', line_stripped)
        if is_roman:
            cleaned_lines.append("<POEM>")
            continue

        # Skip lines with mixed Roman numerals and other capital letters
        contains_roman = re.search(r'\b[IVXLCDM]+\b', line_stripped)
        contains_upper = re.search(r'[A-Z]', line_stripped)
        if contains_roman and contains_upper and not is_roman:
            continue

        # Titles in all caps
        if line_stripped.isupper() and len(line_stripped) > 1:
            cleaned_lines.append(f"<TITLE> {line_stripped} </TITLE>")
        else:
            cleaned_lines.append(line_stripped)

    # Join lines with <NL> token and handle stanzas
    text = "<NL>".join(cleaned_lines)
    text = text.replace("<NL><NL>", "<STANZA>")
    text = text.replace("<STANZA><STANZA><STANZA>", "")

    # ------------------ Encode using tokenizer ------------------
    if tokenizer is None:
        raise ValueError("Please provide a fitted tokenizer")
    token_ids = tokenizer.encode(text)

    # ------------------ Split train/val/test ------------------
    total_len = len(token_ids)
    test_len = int(total_len * test_frac)
    val_len = int(total_len * val_frac)
    train_len = total_len - val_len - test_len

    train_tokens = token_ids[:train_len]
    val_tokens = token_ids[train_len:train_len + val_len]
    test_tokens = token_ids[train_len + val_len:]

    # ------------------ Determine stride for GPT models ------------------
    model_type = config.get('type') if config else None
    if model_type in ("GPT", "GPT2"):
        stride = block_size // 4  # 75% overlap
        print(f"\nUsing overlapping chunks for GPT with stride={stride} (75% overlap).")
    else:
        stride = block_size
        print(f"\nUsing non-overlapping chunks for {model_type}.")

    # ------------------ Convert to blocks ------------------
    def to_blocks(tokens, block_size, stride):
        blocks = [tokens[i:i + block_size] for i in range(0, len(tokens) - block_size, stride)]
        return torch.tensor(blocks, dtype=torch.long, device=device)

    datasets = {
        'train': to_blocks(train_tokens, block_size, stride=stride),
        'val': to_blocks(val_tokens, block_size, stride=block_size),
        'test': to_blocks(test_tokens, block_size, stride=block_size)
    }
    return datasets


def write_eval(csv_file, csv_columns, epoch, train_loss, val_loss, metrics, score):
    """
    Append evaluation metrics for one epoch to a CSV file.

    Args:
        csv_file (str): Path to CSV file.
        csv_columns (list): List of column names.
        epoch (int): Epoch index.
        train_loss (float): Training loss.
        val_loss (float): Validation loss.
        metrics (dict): Metrics dictionary.
        score (float): Combined score.
    """
    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        # Reference-free metrics
        "spelling_accuracy": metrics.get("spelling_accuracy"),
        "unique_correct_frac": metrics.get("unique_correct_frac"),
        # Reference-based metrics
        "train_2-gram_overlap": metrics.get("train_2-gram_overlap"),
        "val_2-gram_overlap": metrics.get("val_2-gram_overlap"),
        "train_3-gram_overlap": metrics.get("train_3-gram_overlap"),
        "val_3-gram_overlap": metrics.get("val_3-gram_overlap"),
        "train_rouge-l": metrics.get("train_rouge-l"),
        "val_rouge-l": metrics.get("val_rouge-l"),
        "train_blank_line_fraction_score": metrics.get("train_blank_line_fraction_score"),
        "val_blank_line_fraction_score": metrics.get("val_blank_line_fraction_score"),
        "train_line_length_mean_score": metrics.get("train_line_length_mean_score"),
        "val_line_length_mean_score": metrics.get("val_line_length_mean_score"),
        "combined_score": score
    }

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writerow(row)


def save_generated_text(generated, model_config, lr, tok_type):
    """
    Save generated text to a file with model and training info in the filename.

    Args:
        generated (str): Generated text content.
        model_config (dict): Model configuration dict.
        lr (float): Learning rate.
        tok_type (str): Tokenizer type.
    """
    output_folder = "generated_text"
    os.makedirs(output_folder, exist_ok=True)

    filename = (
        f"generated_text_{model_config['type']}_em{model_config.get('em_dim','NA')}_"
        f"hidden{model_config.get('hidden_size','NA')}_dmodel{model_config.get('d_model','NA')}_"
        f"dff{model_config.get('d_ff','NA')}_n_heads{model_config.get('n_heads','NA')}_"
        f"layers{model_config.get('n_layers','NA')}_lr{lr}_"
        f"decay{model_config.get('weight_decay','NA')}_"
        f"dropout{model_config.get('dropout','NA')}_toktype{tok_type}.txt"
    )

    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated)

    print(f"\nGenerated text saved to {output_path}")