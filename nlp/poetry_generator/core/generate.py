import torch
import torch.nn.functional as F

from models.rnn import RNN
from models.lstm import LSTM
from models.gpt import GPT

# ==========================================================
#  Text Generation
# ==========================================================
def generate_texts(
    model,
    model_type,
    tokenizer,
    n_generate=100,
    temp=0.9,
    top_k=110,
    top_p=0.9,
    device="cpu",
    num_texts=10,
    eos_token_id=None,
    no_repeat_ngram_size=5,
    repetition_penalty=None,
    bad_words_ids=None,  # list[list[int]] or None
):
    """
    Generate text sequences from a language model (RNN, LSTM, GPT, or GPT-2).

    Args:
        model: PyTorch model (custom RNN/LSTM/GPT or HF GPT-2)
        model_type: "RNN" | "LSTM" | "GPT" | "GPT2"
        tokenizer: has .encode/.decode 
        prefixes: list[str] prompts; defaults to ['<POEM>'] * num_return_sequences
        n_generate: number of new tokens to generate
        temp: temperature
        top_k: top-k cutoff (0 disables)
        top_p: top-p threshold (1.0 disables)
        device: 'cpu' or 'cuda'
        num_texts: how many texts to be generated in parallel
        eos_token_id: optional EOS for early stopping (GPT-2 path)
        no_repeat_ngram_size: optional int (GPT-2 path)
        repetition_penalty: optional float (GPT-2 path)
        bad_words_ids: optional list of token sequences to ban (GPT-2 path)

    Returns:
        list[str]: generated continuations (prompt removed for GPT-2/HF path).
    """

    model.eval()
    
    # ---------------- GPT-2 Path: Use HF built-in generate function ----------------
    if model_type == "GPT2":
        prefix = "<POEM>"
        
        input_indices = torch.tensor([tokenizer.encode(prefix)]).to(device)
        generated_indices = model.generate(
            input_ids=input_indices,
            max_new_tokens=n_generate,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_texts,
            eos_token_id=eos_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
        )
            
    # ---------------- Other Path: Use custom generation process ----------------
    else:
        prefixes = ["<POEM>"] * num_texts
        generated_indices = [tokenizer.encode(p) for p in prefixes]
        generated_tensor = torch.tensor(generated_indices, device=device)
    
        # Initialize hidden state (for RNN/LSTM)
        hidden = None
    
        with torch.no_grad():
            for step in range(n_generate):
                # ---------------- Forward pass ----------------
                if isinstance(model, (RNN, LSTM)):
                    # Feed only the last token and hidden state
                    last_token = generated_tensor[:, -1:].clone()  # [B, 1]
                    logits, hidden = model(last_token, hidden)     # [B, 1, V]
                    logits = logits[:, -1, :]                      # [B, V]
                elif isinstance(model, GPT):
                    # Feed the full sequence for GPT
                    logits = model(generated_tensor)               # [B, seq_len, V]
                    logits = logits[:, -1, :]                      # [B, V]
                else:
                    # HuggingFace GPT-2
                    logits = model(generated_tensor).logits        # [B, seq_len, V]
                    logits = logits[:, -1, :]                      # [B, V]
    
                # ---------------- Temperature scaling ----------------
                logits = logits / temp
                probs = F.softmax(logits, dim=-1)
    
                # ---------------- Top-k & Top-p (nucleus) filtering ----------------
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
                if top_k > 0:
                    sorted_probs[:, top_k:] = 0.0
    
                if top_p < 1.0:
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
    
                # Renormalize probabilities after filtering
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    
                # ---------------- Sample next token ----------------
                next_idx_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_tokens = sorted_indices.gather(-1, next_idx_in_sorted)  # [B, 1]
    
                # Append next token to sequences
                generated_tensor = torch.cat([generated_tensor, next_tokens], dim=1)
                for i in range(len(prefixes)):
                    generated_indices[i].append(next_tokens[i].item())
    
    # ---------------- Decode token sequences ----------------
    generated_texts = [tokenizer.decode(seq) for seq in generated_indices]
    return generated_texts