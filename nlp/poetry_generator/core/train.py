import torch
import csv
import os
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.rnn import RNN
from models.lstm import LSTM
from models.gpt import GPT
from models.gpt2 import GPT2Wrapper
from utils.helpers import set_seed, create_loader, write_eval
from .evaluate import evaluate

# ==========================================================
#  Training
# ==========================================================
def train(
    model_config,
    lr,
    n_epochs,
    batch_size,
    datasets,
    device,
    n_generate,
    tokenizer,
    tok_type,
    label_smoothing=0.1,
    print_every=1,
    eval_every=1,
    base_dir="training_results"
):
    """
    Train RNN, LSTM, GPT, or GPT-2 (optionally with LoRA) and log metrics.
    
    Args:
        model_config: dict containing model hyperparameters and type.
        lr: learning rate.
        n_epochs: number of training epochs.
        batch_size: batch size for training.
        datasets: dict with 'train', 'val', 'test' tensors.
        device: torch device.
        n_generate: number of tokens to generate for evaluation.
        tokenizer: tokenizer wrapper (Char/BPE/GPT2).
        tok_type: string identifier of tokenizer type.
        label_smoothing: label smoothing for CrossEntropyLoss.
        print_every: print interval.
        eval_every: evaluation interval (epochs).
        base_dir: base folder for saving results.
    
    Returns:
        Trained model (best on validation combined score).
    """
    set_seed(42)

    # ------------------ Folder and CSV setup ------------------
    folder_name = (
        f"{model_config['type']}_"
        f"em{model_config.get('em_dim','NA')}_"
        f"hidden{model_config.get('hidden_size','NA')}_"
        f"dmodel{model_config.get('d_model','NA')}_"
        f"n_heads{model_config.get('n_heads','NA')}_"
        f"layers{model_config.get('n_layers','NA')}_"
        f"lr{lr}_"
        f"decay{model_config.get('weight_decay','NA')}_"
        f"dropout{model_config.get('dropout','NA')}_"
        f"toktype{tok_type}"
    )
    
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_file = os.path.join(save_dir, "metrics.csv")

    
    csv_columns = [
        "epoch", "train_loss", "val_loss",
        "spelling_accuracy",
        "unique_correct_frac",
        "train_2-gram_overlap", "val_2-gram_overlap",
        "train_3-gram_overlap", "val_3-gram_overlap",
        "train_rouge-l", "val_rouge-l",
        "train_blank_line_fraction_score", "val_blank_line_fraction_score",
        "train_line_length_mean_score", "val_line_length_mean_score",
        "combined_score"
    ]
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # ------------------ Metric weights ------------------
    metric_weights = {
        "spelling_accuracy": 0.1,
        "unique_correct_frac": 0.1,
        "val_2-gram_overlap": 10 * 0.1,
        "val_3-gram_overlap": 0.0,
        "val_rouge-l": 0.1,
        "val_blank_line_fraction_score": 0.0,
        "val_line_length_mean_score": 0.1
    }

    # ------------------ Model selection ------------------
    vocab_size = max(datasets['train'].max().item(),
                     datasets['val'].max().item(),
                     datasets['test'].max().item()) + 1
    
    if model_config['type'] == 'RNN':
        model = RNN(vocab_size=vocab_size,
                    em_dim=model_config['em_dim'],
                    hidden_size=model_config['hidden_size'],
                    n_layers=model_config['n_layers'],
                    dropout=model_config['dropout'])
        
    elif model_config['type'] == 'LSTM':
        model = LSTM(vocab_size=vocab_size,
                     em_dim=model_config['em_dim'],
                     hidden_size=model_config['hidden_size'],
                     n_layers=model_config['n_layers'],
                     dropout=model_config['dropout'])
        
    elif model_config['type'] == 'GPT':
        model = GPT(vocab_size=vocab_size,
                    d_model=model_config['d_model'],
                    n_heads=model_config['n_heads'],
                    n_layers=model_config['n_layers'],
                    dropout=model_config['dropout'],
                    bias=model_config['bias'])
        
    elif model_config['type'] == 'GPT2':
        wrapper = GPT2Wrapper(tokenizer, model_config)
        model = wrapper.model

    model.to(device)
    print(f"\nStart training {model_config} for {n_epochs} epochs with batch size {batch_size}, learning rate {lr}, tokenizer={tok_type}...\n")

    # ------------------ DataLoaders ------------------
    is_gpt = model_config['type'] in ['GPT', 'GPT2']
    train_loader = create_loader(datasets['train'], batch_size, shuffle=not is_gpt)
    val_loader = create_loader(datasets['val'], batch_size, shuffle=False)

    # ------------------ Criterion, optimizer, scheduler ------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen params entirely
    
        # Exclude biases, LayerNorms, and embeddings from weight decay
        if (
            "bias" in name
            or "ln" in name.lower()
            or "norm" in name.lower()
            or "wte" in name  # GPT-2 input embeddings
            or "lm_head" in name  # tied output head (same weights as wte)
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": model_config.get("weight_decay", 0.01)},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

    # ------------------ Keep track of best model ------------------
    best_score = -float('inf')
    combined_score = 0.0

    # ------------------ Pre-training evaluation (baseline) ------------------
    model.eval()
    
    # Compute train loss
    total_train_loss = 0
    with torch.no_grad():
        for batch in train_loader:
            x_batch = batch[0][:, :-1].to(device)
            y_batch = batch[0][:, 1:].to(device)
            outputs = model(x_batch)
            logits = (
                outputs.logits
                if model_config['type'] == 'GPT2'
                else (outputs[0] if isinstance(outputs, tuple) else outputs)
            )
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.reshape(-1))
            total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_loader)
    
    # Compute val loss
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_val = batch[0][:, :-1].to(device)
            y_val = batch[0][:, 1:].to(device)
            outputs = model(x_val)
            logits = (
                outputs.logits
                if model_config['type'] == 'GPT2'
                else (outputs[0] if isinstance(outputs, tuple) else outputs)
            )
            loss = criterion(logits.view(-1, logits.size(-1)), y_val.reshape(-1))
            total_val_loss += loss.item()
    val_loss = total_val_loss / len(val_loader)
    
    # Compute generation metrics
    metrics = evaluate(
        model,
        model_config['type'],
        tokenizer,
        tok_type,
        train_dataset=datasets['train'],
        val_dataset=datasets['val'],
        n_generate=n_generate,
        device=device
    )
    
    # --- Write to CSV ---
    write_eval(csv_file, csv_columns, -1, train_loss, val_loss, metrics, combined_score)
    
    print(f"Pre-training evaluation done. Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    
    # ------------------ Training loop ------------------
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x_batch = batch[0][:, :-1].to(device)
            y_batch = batch[0][:, 1:].to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            logits = outputs.logits if model_config['type']=='GPT2' else (outputs[0] if isinstance(outputs, tuple) else outputs)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_val = batch[0][:, :-1].to(device)
                y_val = batch[0][:, 1:].to(device)
                outputs = model(x_val)
                logits = outputs.logits if model_config['type']=='GPT2' else (outputs[0] if isinstance(outputs, tuple) else outputs)
                val_loss += criterion(logits.view(-1, logits.size(-1)), y_val.reshape(-1)).item()
        val_loss /= len(val_loader)

        # ------------------ Evaluation metrics ------------------
        if (epoch + 1) % eval_every == 0:
            metrics = evaluate(
                model, model_config['type'], tokenizer, tok_type,
                train_dataset=datasets['train'], val_dataset=datasets['val'],
                n_generate=n_generate, device=device
            )

            # Combined score
            combined_score = 0.0
            for key, weight in metric_weights.items():
                combined_score += weight * metrics.get(key, 0.0)

            write_eval(csv_file, csv_columns, epoch, train_loss, val_loss, metrics, combined_score)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Combined Score: {combined_score:.4f}")

        # ------------------ Save best model ------------------
        if combined_score > best_score:
            best_score = combined_score
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    # ------------------ Load best model ------------------
    best_model_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"Training finished. Best model loaded from {save_dir}")

    return model