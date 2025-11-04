import re
import enchant

from .generate import generate_texts

# ==========================================================
#  Reference-free metrics
# ==========================================================
def evaluate_spelling_accuracy(generated):
    """
    Compute the fraction of correctly spelled words in a generated text.
    
    Args:
        generated: The generated text string.
        
    Returns:
        A float in [0, 1] representing spelling accuracy.
    """
    checker = enchant.Dict("en_US")
    words = re.findall(r"[A-Za-z]+", generated.lower())
    if not words:
        return 0.0
    correct_total = sum(1 for w in words if checker.check(w))
    return correct_total / len(words)


def evaluate_diversity(generated):
    """
    Compute the fraction of unique correctly spelled words in a generated text.
    
    Args:
        generated: The generated text string.
        
    Returns:
        A float in [0, 1] representing vocabulary diversity.
    """
    checker = enchant.Dict("en_US")
    words = re.findall(r"[A-Za-z]+", generated.lower())
    unique_words = set(words)
    if not unique_words:
        return 0.0
    correct_words = {w for w in unique_words if checker.check(w)}
    return len(correct_words) / len(words)


# ==========================================================
#  Reference-based metrics
# ==========================================================
def evaluate_overlap(reference, generated):
    """
    Compute n-gram overlap (2-grams and 3-grams) between reference and generated text.
    
    Args:
        reference: The reference text string.
        generated: The generated text string.
        
    Returns:
        Dictionary with keys '2' and '3' representing overlap fractions.
    """
    def get_ngrams(tokens, n):
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    gen_words = re.findall(r"[A-Za-z]+", generated)
    ref_words = re.findall(r"[A-Za-z]+", reference)
    
    overlaps = {}
    for n in (2, 3):
        gen_ngrams = set(get_ngrams(gen_words, n))
        ref_ngrams = set(get_ngrams(ref_words, n))
        overlaps[f"{n}"] = len(gen_ngrams & ref_ngrams) / len(gen_ngrams) if gen_ngrams else 0.0
    return overlaps


def evaluate_rouge_l_score(reference, generated):
    """
    Compute ROUGE-L score between reference and generated text (precision-only).
    
    Args:
        reference: Reference text string.
        generated: Generated text string.
        
    Returns:
        Float ROUGE-L score in [0, 1].
    """
    ref_words = reference.split()
    gen_words = generated.split()
    m, n = len(ref_words), len(gen_words)

    lcs_table = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_words[i-1] == gen_words[j-1]:
                lcs_table[i][j] = lcs_table[i-1][j-1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i-1][j], lcs_table[i][j-1])

    lcs_len = lcs_table[m][n]
    return lcs_len / n


def evaluate_structure(reference, generated):
    """
    Evaluate structural similarity between generated and reference poems.
    Measures blank-line fraction (fraction of stanzas) and mean line length similarity.
    
    Args:
        reference: Reference poem text.
        generated: Generated poem text.
    
    Returns:
        Tuple (blank_line_fraction_score, line_length_mean_score) in [0, 1].
    """
    def split_poems(text):
        poems, current = [], []
        blank_counter = 0
        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.isupper() and len(stripped) > 1:
                if current:
                    poems.append("\n".join(current).strip())
                    current = []
                current.append(line)
                blank_counter = 0
            elif not stripped:
                blank_counter += 1
                if blank_counter >= 5 and current:
                    poems.append("\n".join(current))
                    current = []
                    blank_counter = 0
                current.append(line)
            else:
                blank_counter = 0
                current.append(line)
        if current:
            poems.append("\n".join(current).strip())
        return [p for p in poems if p.strip()]

    def fraction_blank_lines(text):
        poems = split_poems(text)
        fractions = []
        for poem in poems:
            lines = poem.splitlines()
            if lines:
                blank = sum(1 for line in lines if not line.strip())
                fractions.append(blank / len(lines))
        return sum(fractions)/len(fractions) if fractions else 0.0

    def mean_line_length(text):
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0
        return sum(len(line) for line in lines) / len(lines)

    gen_blank_frac = fraction_blank_lines(generated)
    ref_blank_frac = fraction_blank_lines(reference)
    gen_mean_len = mean_line_length(generated)
    ref_mean_len = mean_line_length(reference)

    # --- Turn into scores ---
    blank_score = max(0.0, 1.0 - abs(gen_blank_frac - ref_blank_frac)/ref_blank_frac) if ref_blank_frac > 0 else 0.0
    line_len_score = max(0.0, 1.0 - abs(gen_mean_len - ref_mean_len)/ref_mean_len) if ref_mean_len > 0 else 0.0
    return blank_score, line_len_score


# ==========================================================
#  Main evaluation function
# ==========================================================
def evaluate(model,
             model_type,
             tokenizer,
             tok_type,
             train_dataset,
             val_dataset,
             n_generate,
             device='cpu'):
    """
    Evaluate generated text quality on reference-free and reference-based metrics for 10 generated texts.

    Args:
        model: PyTorch model (RNN, LSTM, GPT, GPT2).
        model_type: 'GPT2' or other.
        tokenizer: Tokenizer instance compatible with model.
        tok_type: 'char' or 'bpe'.
        train_dataset: Tensor of training data for reference metrics.
        val_dataset: Tensor of validation data for reference metrics.
        n_generate: Maximum number of tokens to generate.
        device: Device for model evaluation.
        prefixes: List of prefix strings for generation.

    Returns:
        Dictionary with averaged metrics over all generated texts.
    """
    model.eval()

    # --- Generate texts ---
    top_k = 20 if tok_type=='char' else 110
    generated_texts = generate_texts(model, model_type, tokenizer, n_generate=n_generate, top_k=top_k, device=device)

    # --- Prepare reference texts ---
    train_ref_text = tokenizer.decode(train_dataset.reshape(-1).tolist())
    val_ref_text = tokenizer.decode(val_dataset.reshape(-1).tolist())

    # --- Compute metrics ---
    metrics_batch = []
    for gen_text in generated_texts:
        metrics = {}
        # Reference-free
        metrics["spelling_accuracy"] = evaluate_spelling_accuracy(gen_text)
        metrics["unique_correct_frac"] = evaluate_diversity(gen_text)
        # Reference-based train
        train_overlaps = evaluate_overlap(train_ref_text, gen_text)
        for n in (2, 3):
            metrics[f"train_{n}-gram_overlap"] = train_overlaps[f"{n}"]
        metrics["train_rouge-l"] = evaluate_rouge_l_score(train_ref_text, gen_text)
        metrics["train_blank_line_fraction_score"], metrics["train_line_length_mean_score"] = evaluate_structure(train_ref_text, gen_text)
        # Reference-based val
        val_overlaps = evaluate_overlap(val_ref_text, gen_text)
        for n in (2, 3):
            metrics[f"val_{n}-gram_overlap"] = val_overlaps[f"{n}"]
        metrics["val_rouge-l"] = evaluate_rouge_l_score(val_ref_text, gen_text)
        metrics["val_blank_line_fraction_score"], metrics["val_line_length_mean_score"] = evaluate_structure(val_ref_text, gen_text)
        metrics_batch.append(metrics)

    # --- Average metrics ---
    avg_metrics = {key: sum(m[key] for m in metrics_batch)/len(metrics_batch) for key in metrics_batch[0].keys()}
    return avg_metrics