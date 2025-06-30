import matplotlib.pyplot as plt
import json
import numpy as np

from data_process import *

# Helper functions to plot the final ETM-LDA comparison we are looking for

# Save results to a JSON file
def save_results_to_json(results, file_name="results.json"):
    with open(file_name, "w") as file:
        json.dump(results, file)
    print(f"Results saved to {file_name}")


# Load results from JSON file
def load_results_from_json(file_name="results.json"):
    with open(file_name, "r") as file:
        results = json.load(file)
    print(f"Results loaded from {file_name}")
    return results


# Plot the ELBO values
def plot_elbo(model):
      plt.figure(figsize=(10, 6))
      plt.plot(model.elbos, label='Loss', color='blue')

      plt.xlabel('Epochs', fontsize=12)
      plt.ylabel('Loss', fontsize=12)
      plt.title('Loss vs Epochs', fontsize=14)
      plt.legend(fontsize=12)

      plt.grid(True, linestyle='--', alpha=0.6)
      plt.show()
      

# Plot the Results
def plot_results(results):
    styles = {
        "ETM": {"color": "tab:blue", "marker": "o"},
        "LDA": {"color": "tab:orange", "marker": "s"}
    }

    # Create subplots: one per vocabulary size
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5), sharey=True)

    # Iterate through each vocabulary size and its corresponding model results
    for ax, (vocab_size, model_results) in zip(axes, results.items()):
        for model, metrics in model_results.items():
            ax.scatter(metrics["predictive_power"],
                       metrics["interpretability"],
                       color=styles[model]["color"],
                       marker=styles[model]["marker"],
                       s=100,
                       label=model)
        ax.set_title(vocab_size)
        ax.set_xlabel("Predictive Power")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)  # horizontal reference line
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)  # vertical reference line

    axes[0].set_ylabel("Interpretability")  # shared y axis (interpretability)

    # Legend
    handles = [plt.Line2D([0], [0], marker=styles[model]["marker"], color='w',
                          markerfacecolor=styles[model]["color"], markersize=10, label=model)
               for model in styles.keys()]
    fig.legend(handles=handles, loc="upper right", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust the layout to prevent overlappings
    plt.show()



# Helper Functions to calculate topic diversity and coherence

# Compute topic diversity given the topic-word distribution matrix beta.
# topic diversity is the percentage of unique words in the top 25 words of all topics. Diversity close to 0 indicates redundant topics; diversity close to 1 indicates more varied topics.
def get_topic_diversity (beta, topk):
    # beta: topic-word distribution matrix (topics_size x num_words)
    # topk: number of top words to consider per topic

    topics_size = beta.shape[0]  # Number of topics
    list_w = np.zeros((topics_size, topk))

    # Get the top-k words for each topic
    for k in range(topics_size):
        idx = beta[k, :].argsort()[-topk:][::-1]  # Get indices of top-k words
        list_w[k, :] = idx

    # Calculate the proportion of unique words across all topics
    n_unique = len(np.unique(list_w))  # Count unique words
    TD = n_unique / (topk * topics_size)  # Proportion of unique words

    return TD


# Compute topic coherence given the topic-word distribution matrix beta.
# Topic coherence is the average pointwise mutual information of two words drawn randomly from the same document (see formula in the paper)
# Most likely words in a coherent topic should have high mutual information.
def get_topic_coherence(beta, data, vocab):
    # beta (np.ndarray): The topic-word distribution matrix of size (topics_size x vocab_size).
    # data (list of lists): Preprocessed test documents as a list of word indices.
    # vocab (list): The vocabulary list where index corresponds to the word.


    def get_document_frequency(data, word_idx, co_word_idx=None):   # get document frequency for a word or pair of words
        D_word = 0
        D_co_word = 0
        D_word_co_word = 0

        for doc in data:
            #print("0")
            #print(word_idx)
            #print(doc)
            if doc[word_idx] > 0:
                #print("1")
                D_word += 1
            if co_word_idx is not None and doc[co_word_idx] > 0:
                #print("2")
                D_co_word += 1
                if doc[word_idx] > 0:
                    #print("3")
                    D_word_co_word += 1

        if co_word_idx is None:
            return D_word
        return D_co_word, D_word_co_word


    D = len(data)  # Total number of documents, data is list of documents
    topics_size = beta.shape[0]  # Number of topics
    topic_coherence = []

    for k in range(topics_size):
        # Get the top 10 words for the current topic
        top_words_indices = beta[k, :].argsort()[-10:][::-1]  # Indices of top-10 words
        top_words = [vocab[idx] for idx in top_words_indices]
        TC_k = 0
        counter = 0 # comment this if doing it like in the github

        for i, word_idx in enumerate(top_words_indices):
            # Get document frequency for the word
            D_word = get_document_frequency(data, word_idx)

            for j in range(i + 1, len(top_words_indices)):
                co_word_idx = top_words_indices[j]

                # Get document frequency for the word pair
                D_co_word, D_word_co_word = get_document_frequency(data, word_idx, co_word_idx)

                # Compute pairwise coherence
                if D_word_co_word == 0:
                    f_wi_wj = -1
                else:
                    #print("Inside else of D_word_co_word")
                    f_wi_wj = -1 + (np.log(D_word) + np.log(D_co_word) - 2.0 * np.log(D)) / \
                                      (np.log(D_word_co_word) - np.log(D))

                TC_k += f_wi_wj
                counter += 1

        # Add coherence score for the topic
        topic_coherence.append(TC_k / counter if counter > 0 else 0.0)  # comment this to do it like in the github
        # topic_coherence.append(TC_k) # this way to do it like in the github

    TC = np.mean(topic_coherence)
    # this way to do it like in the github:
    # if counter ==0: retrn 0.0
    # TC = np.sum(topic_coherence) / counter
    print(f"Topic Coherence: {TC:.4f}")
    return TC

