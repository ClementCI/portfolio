import torch

from utils.helpers import extract_config, get_tokenizer, get_datasets, save_generated_text
from config import config_list
from core.train import train
from core.generate import generate_texts

# ==========================================================
#  Main
# ==========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for config in config_list:
        # ------------------ Extract configuration ------------------
        model_config, lr, n_epochs, batch_size, block_size, print_every, eval_every, n_generate, tok_type = extract_config(config)
        
        # ------------------ Get fitted tokenizer ------------------
        tokenizer = get_tokenizer(book_fname='data/dickinson.txt', tok_type=tok_type)
        
        # ------------------ Get encoded datasets ------------------
        datasets = get_datasets(book_fname='data/dickinson.txt', device=device, tokenizer=tokenizer, block_size=block_size, config=model_config)
        
        # ------------------ Train ------------------
        model = train(model_config=model_config, 
                      lr=lr,
                      n_epochs=n_epochs, 
                      batch_size=batch_size,
                      datasets=datasets, 
                      device=device, 
                      n_generate=n_generate,
                      tokenizer=tokenizer,
                      tok_type=tok_type,
                      print_every=print_every,
                      eval_every=eval_every)
    
        # ------------------ Generate and save ------------------
        top_k = 20 if tok_type=='char' else 110
        generated = generate_texts(model=model,
                               model_type=model_config['type'],
                               tokenizer=tokenizer, 
                               n_generate=700,
                               top_k=top_k, 
                               device=device)[0]
        save_generated_text(generated, model_config, lr, tok_type)


if __name__=="__main__":
    main()