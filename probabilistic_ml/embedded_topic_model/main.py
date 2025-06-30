import numpy as np
from torch import nn

from data_process import *
from model import *
from utils import *

## Parameters
# Model parameters - These are taken from the default paper
topics_size = 100    # number of topics
vocabulary_size = len(vocab)    # length of the vocabulary
hidden_size = 800    # hidden space dimensions
em_size = 300    # dimensions of rho
embedding_size = 300    # dimensions of embedding
theta_activation = nn.ReLU()    # Popular NN activation; others like tanh or leaky ReLU also possible

# Optimization/Training parameters
num_of_epochs = 150  # number of training epochs
learning_rate = 0.001
weight_decay = 0.001

# Data, Logging, and visualization parameters
batch_size = 100
log_interval = 10

if __name__ == "__main__":
    ## Run the model
    # Initialize the model
    model = ETM(topics_size, vocabulary_size, hidden_size, em_size, theta_activation)
    model.set_optimizer(learning_rate, weight_decay)

    ## Training
    best_epoch = 0
    best_val = np.inf
    all_val = []

    # Dictionary to save interpretability and predictive power for each vocabulary size
    results = {}

    for epoch in range(num_of_epochs):
        print(f"NEW EPOCH ------> {epoch}")

        # Perform the training step and evaluate
        model.train_epoch(training_set, batch_size, log_interval)
        print("=" * 100)
        print(f"Evaluation of epoch {epoch}")
        cur_val = model.evaluate(training_set, vocab, test_1, test_2, batch_size)
        print("=" * 100)

        # Update best val if better and save model
        if cur_val[0] < best_val:
            best_epoch = epoch
            best_val = cur_val[0]

        # Potentially visualize
        all_val.append(cur_val[0])  # keep track of all the val

    # Print training logs
    print(model.elbos)

    print("=" * 100)
    print("Training finished, doing last eval now")

    results = model.evaluate(training_set, vocab, test_1, test_2, batch_size, tc=True, td=True)

    print("Perplexity = ", results[0])
    print("Predictive Power = ", results[1])
    print("Interpretability = ", results[2])
    print("TC = ", results[3])
    print("TD = ", results[4])
    
    # Save model and results
    save_results_to_json(results)
    file_name_model = f"{path}_etm_model.pth"
    torch.save(model, file_name_model)
