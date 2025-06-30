import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch import nn, optim

from utils import *
from data_process import *

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)


# Model
class ETM(nn.Module):
    def __init__(self, topics_size, vocab_size, hidden_size, em_size, theta_act, drop_rate=0.5):
        super(ETM, self).__init__()

        ## Hyperparameters
        self.topics_size = topics_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.em_size = em_size
        self.theta_act = theta_act
        self.drop_rate = drop_rate
        self.drop = nn.Dropout(drop_rate)

        # Initialize optimize
        self.optimizer = None

        # ELBO for visualization
        self.elbos = []

        ## Word embedding matrix rho
        self.rho = nn.Linear(em_size, vocab_size, bias=False)

        ## Topic embedding matrix alpha
        self.alphas = nn.Linear(em_size, topics_size, bias=False)

        ## Variational distribution for theta
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, hidden_size),
                self.theta_act,
                nn.Linear(hidden_size, hidden_size),
                self.theta_act,
            )

        # Variational parameters
        self.mu_theta = nn.Linear(hidden_size, topics_size, bias=True)
        self.log_sigma_theta = nn.Linear(hidden_size, topics_size, bias=True)

    ## Set_optimizer - to set the optimizer after creating the model
    def set_optimizer(self, lr, weight_decay):
      self.optimizer = optim.Adam(self.parameters(), lr=lr)

    ## Encode - function responsible for encoding part, should return parameters of the variational distribution for \theta.
    def encode(self, batch):
      # variational distribution
      q_theta = self.q_theta(batch)
      q_theta = self.drop(q_theta) # apply dropout
      # variational parameters from NN
      mu_theta = self.mu_theta(q_theta)
      log_sigma_theta = self.log_sigma_theta(q_theta)

      log_sigma_theta = torch.clamp(log_sigma_theta, min=-1, max=1)

      return mu_theta, log_sigma_theta

    ## Decode - function responsible for the decoding part, compute the probability of topic given the document
    def decode(self, theta, beta):
      res = torch.matmul(theta, beta)
      pred = torch.log(res + 1e-6) # for numerical stability
      return pred

    ## Compute_beta - generate the description as a definition over words= generates the topic-word distribution matrix beta (topics_size x vocab_size) each row represents a topic
    def compute_beta(self):
      logit = self.alphas(self.rho.weight)  # project the word embeddings into the topic space using the linear transformation 'alphas' (vocab_size x topics_size)
      beta = F.softmax(logit.T, dim=-1)     # apply softmax to normalise over the vocabulary dimension (topics_size x vocab_size)
      return beta                           # the values in the beta matrix are probabilities indicating how likely each work is to belong to a given topic

    ## Compute_theta - getting the topic proportion for the document passed in the normalize bow or tf-idf
    def compute_theta(self, normalized_batch):
      # Get parameters from NN
      mu_theta, log_sigma_theta = self.encode(normalized_batch)

      # Sample delta with reparameterization trick
      if self.training: # check if the model is in training mode
          std = torch.exp(0.5 * log_sigma_theta)
          eps = torch.randn_like(std)
          delta_sample = mu_theta + eps * std
      else: # if evaluation mode
          delta_sample = mu_theta # use only the mean for deterministic behavior

      # Compute theta
      theta = F.softmax(delta_sample, dim=-1)
      return theta

    ## Forward
    def forward(self, batch, normalized_batch):
      beta = self.compute_beta()
      theta = self.compute_theta(normalized_batch)
      mu_theta, log_sigma_theta = self.encode(batch)

      # Reconstruction loss
      pred = self.decode(theta, beta)
      recon_loss = -(pred * batch).sum(1)

      # KL divergence
      kl = -0.5 * torch.sum(1 + log_sigma_theta - mu_theta.pow(2) - torch.exp(log_sigma_theta), dim=-1).mean()
      return recon_loss, kl

    # Plot the ELBO values
    def plot_elbo(self):
      plt.figure(figsize=(10, 6))
      plt.plot(self.elbos, label='ELBO', color='blue')

      plt.xlabel('Epochs', fontsize=12)
      plt.ylabel('ELBO', fontsize=12)
      plt.title('ELBO vs Epochs', fontsize=14)
      plt.legend(fontsize=12)

      plt.grid(True, linestyle='--', alpha=0.6)
      plt.show()

    ## Train_epoch - encapsulates the training logic, ensuring that the model learns from data while keeping track of performance metrics
    def train_epoch(self, training_set, batch_size, log_interval):
      self.train() # Set the model to training mode
      acc_loss = 0
      acc_kl_loss = 0
      count = 0

      # Randomize the data for batches, generate random order of indices and split
      random_perm = torch.randperm(num_of_docs_train)
      batch_indices = torch.split(random_perm, batch_size)

      for idx, batch in enumerate(batch_indices):
        # Reset the gradients from previous epoch
        self.optimizer.zero_grad()
        self.zero_grad()

        # Get the current batch
        current_batch = torch.from_numpy(training_set[batch, :]).float().to(device)

        # Normalize the current batch
        sum = current_batch.sum(1).unsqueeze(1)  # Sum word counts for each document
        normalized_current_batch = current_batch / sum  # Normalize word counts

        # Perform the forward step and calculate the error
        reconstruction_loss, kl_loss = self.forward(current_batch, normalized_current_batch)
        total_loss = reconstruction_loss + kl_loss
        total_loss = total_loss.mean()    # TODO: Before it's an array of values
        total_loss.backward()

        self.optimizer.step()

        acc_loss += torch.sum(reconstruction_loss).item()
        acc_kl_loss += torch.sum(kl_loss).item()
        count += 1

        if idx > 0 and idx % log_interval == 0:
          # Calculate the average loss so far
          cur_real_loss = round(acc_loss / count + acc_kl_loss / count, 3)

          #print("-"*50)
          print(f"Epoch = {epoch}, Batch {idx+1} out of {len(batch_indices)}, Current average loss = {cur_real_loss}")

      # Report final info
      cur_real_loss = round(acc_loss / count + acc_kl_loss / count, 3)
      self.elbos.append(cur_real_loss)    # keep track of the elbo over epochs
      print("-"*100)
      print(f"Epoch {epoch} finished, the average loss is {cur_real_loss}")
      # print(f"Scheduler Step: Epoch {epoch}, LR: {self.scheduler.get_last_lr()}")

    ## Visualize: in their code, it contributes to part of table 3 (top five words) and word visualization. But it doesn't even get exactly those tables.


    ## Evaluate: Evaluates the ETM model on test data
    def evaluate(self, training_set, vocabulary, test_1, test_2, batch_size=100, topk=25, tc=False, td=False):
      # training_set: (to calculate topic coherence)
      # vocabulary: list of vocabulary words (to calculate topic coherence)
      # test_1 (torch.Tensor): first half of the test data for inferring topic proportions.
      # test_2 (torch.Tensor): second half of the test data for calculating reconstruction loss.
      # batch_size: batch size for processing the test data. I choose 128, maybe needs changing?
      # opk: number of top words to use for topic coherence and diversity

      # Initialisation
      TC = 1
      TD = 1
      Interpretability = 1

      self.eval()  # Set the model to evaluation mode
      total_loss = 0  # Accumulate total reconstruction loss
      total_words = 0  # Accumulate total words in the second test batch

      # Calculate the number of batches
      # num_batches = num_of_docs_test_1 // batch_size
      indices_1 = torch.split(torch.tensor(range(test_1.shape[0])), batch_size)

      # Disable gradient calculations for efficiency
      with torch.no_grad():
          #for i in range(num_batches):
          for i in range(len(indices_1)):
              # 1. Get the current batch for both test parts
              batch_1 = torch.from_numpy(test_1[i * batch_size: (i + 1) * batch_size]).float().to(device)
              batch_2 = torch.from_numpy(test_2[i * batch_size: (i + 1) * batch_size]).float().to(device)
              #batch_1 = test_1[i * batch_size: (i + 1) * batch_size].to(device)
              #batch_2 = test_2[i * batch_size: (i + 1) * batch_size].to(device)

              # 2. Normalize batch_1 to infer topic proportions (theta)
              sums_1 = batch_1.sum(1).unsqueeze(1)  # Sum word counts for each document
              normalized_batch_1 = batch_1 / sums_1  # Normalize word counts

              # 3. Infer topic proportions theta using compute_theta
              theta = self.compute_theta(normalized_batch_1)

              # 4. Get the topic-word distribution matrix beta
              beta = self.compute_beta()

              # 5. Reconstruct the document using theta and beta
              preds = torch.mm(theta, beta)  # Matrix multiplication: theta (batch_size x topics_size) * beta (topics_size x vocab_size)

              #preds = torch.log(preds + 1e-6)  # Apply log to avoid log(0) by adding a small epsilon
              preds = torch.log(preds)

              # 6. Compute reconstruction loss using batch_2
              recon_loss = -(preds * batch_2).sum(1)  # Element-wise product and sum over words
              sums_2 = batch_2.sum(1).unsqueeze(1)
              words_batch = sums_2.squeeze()

              loss = recon_loss/words_batch
              loss = np.nanmean(loss)
              total_loss += loss

              #print("recon_loss = ", recon_loss)
              #total_loss += recon_loss.sum().item()  # Accumulate total loss
              #print("total_loss = ", total_loss)
              #total_words += batch_2.sum().item()  # Accumulate total word count in batch_2

      # 7. Compute perplexity
      predictive_power = total_loss/ len(indices_1)
      #predictive_power = predictive_power/training_set.sum()
      #avg_loss = total_loss / total_words  # Average negative log-likelihood per word
      perplexity = math.exp(predictive_power)#/len(vocabulary)  # Perplexity = exp(average loss)

      if tc or td:
        beta = self.compute_beta().cpu().detach().numpy()
        if td:
          # 8. Compute topic diversity (TD)
          print("Computing TD...")
          TD = get_topic_diversity(beta, topk = topk)
        if tc:
          # 9. Compute topic coherancy (TC)
          print("Computing TC...")
          TC = get_topic_coherence(beta, training_set, vocabulary)
          #perplexity = perplexity/TC

      interpretability = None
      if tc and td:
        # 10. Compute Topic Quality and Interpretability
        print("Computic TQ and Interpretability")
        TQ = TD * TC
        interpretability = math.exp(TQ)

      # 10. Print and return the perplexity score
      print(f"Perplexity on test data: {perplexity:.2f}")
      return perplexity, -predictive_power, interpretability, TC, TD