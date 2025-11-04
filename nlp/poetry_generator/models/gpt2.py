from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

# ==========================================================
#  GPT-2 Language Model Wrapper
# ==========================================================
class GPT2Wrapper:
    def __init__(self, tokenizer, model_config):
        """
        model_config keys:
          - type: 'GPT2'
          - use_lora: bool (default False)
          - train_layernorm: bool (default True)
          - last_k_layers: int (default 0)  # unfreeze last k transformer blocks
        """
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.model_config["type"] != "GPT2":
            raise ValueError("This class only supports GPT2 models.")

        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Save vocab size before resizing for selective_unfreeze
        self.old_vocab = model.config.vocab_size

        # Resize embeddings to include new special tokens
        model.resize_token_embeddings(self.tokenizer.vocab_size)

        # Apply LoRA (PEFT sets requires_grad on LoRA params automatically)
        if self.model_config.get("use_lora", False):
            model = self.apply_lora(model)

        # Selective unfreezing
        self.selective_unfreeze(
            model,
            train_layernorm=self.model_config.get("train_layernorm", True),
            last_k_layers=self.model_config.get("last_k_layers", 0),
        )

        return model

    def apply_lora(self, model):
        # Target both attention and MLP projections for better style control
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn", "c_fc", "c_proj"],  # GPT-2 module names
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_config)

    def selective_unfreeze(self, model, train_layernorm=True, last_k_layers=0):
        """
        Custom unfreezing logic.

        - Train ONLY the embeddings of newly added special tokens (ids in [old_vocab:]).
        - Optionally unfreeze LayerNorms.
        - Optionally unfreeze the last k transformer blocks.
        - Keep all other parameters frozen (including lm_head).
        - LoRA params stay trainable automatically (set by PEFT if used).
        """
        # 1) Default: freeze everything
        for _, p in model.named_parameters():
            p.requires_grad = False

        # 2) Allow LoRA adapter weights
        for name, p in model.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = True

        # 3) Unfreeze LayerNorms globally if requested
        if train_layernorm:
            for name, p in model.named_parameters():
                if (
                    ".ln_" in name
                    or name.endswith(".ln_f.weight")
                    or name.endswith(".ln_f.bias")
                ):
                    p.requires_grad = True

        # 4) Unfreeze last k transformer blocks if requested
        if last_k_layers and last_k_layers > 0:
            n_layers = getattr(model.config, "n_layer", None)
            if n_layers is None:
                raise ValueError("Model config missing n_layer for GPT-2.")
            start = max(0, n_layers - last_k_layers)

            for name, p in model.named_parameters():
                # names like: transformer.h.{idx}.*
                if name.startswith("transformer.h."):
                    parts = name.split(".")
                    if len(parts) > 2 and parts[2].isdigit():
                        idx = int(parts[2])
                        if idx >= start:
                            p.requires_grad = True

        # 5) Train ONLY the new embedding rows
        emb = model.get_input_embeddings()  # GPT-2: transformer.wte
        old_vocab = self.old_vocab
        new_vocab = emb.weight.shape[0]

        # Clean up previous hook safely
        if hasattr(self, "_emb_grad_hook"):
            try:
                self._emb_grad_hook.remove()
            except Exception:
                pass
            del self._emb_grad_hook

        # If new tokens were added, make only those rows trainable via a grad mask
        if new_vocab > old_vocab:
            emb.weight.requires_grad_(True)

            def grad_mask_for_new_rows(grad):
                """Zero out gradients for old tokens, keep new tokens trainable."""
                mask = grad.new_zeros(grad.shape)
                mask[old_vocab:] = 1.0
                return grad * mask

            self._emb_grad_hook = emb.weight.register_hook(grad_mask_for_new_rows)
        else:
            # No new tokens: keep embeddings frozen
            emb.weight.requires_grad_(False)
