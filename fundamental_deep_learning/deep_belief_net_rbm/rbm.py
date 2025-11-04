from util import *
import numpy as np
import matplotlib.pyplot as plt

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10, period=5000):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.7

        self.print_period = 5000
        
        self.weight_decay = 0.001
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : period, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        self.loss=[] # to monitor reconstrucuction loss
        
        return

        
    def cd1(self,visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]

        for it in range(n_iterations):

            # Positive phase (data distribution)
            minibatch_indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            v_0 = visible_trainset[minibatch_indices,:] # minibatch clamped on training data
            h_0 = self.get_h_given_v(v_0)[-1] # activations

            # Negative phase (model distribution)
            v_1 = self.get_v_given_h(h_0)[0] # probabilities
            h_1 = self.get_h_given_v(v_1)[0] # probabilities for last hidden update
        
            self.update_params(v_0, h_0, v_1, h_1)
            
            # visualize once in a while when visible layer is input images
            
            if it % self.rf["period"] == 0 and self.is_bottom:
                
                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # print progress
            
            if it % self.rf["period"] == 0 or it == n_iterations:
                recon_loss = np.mean(np.linalg.norm(v_0 - v_1, axis=1))
                self.loss.append(recon_loss)
                print ("iteration=%7d recon_loss=%4.4f"%(it, recon_loss))
        
        return
    
    def get_reconstruction_loss(self):
        if not self.loss:
            raise RuntimeError("This RBM is not trained yet.")
        else:
            return self.loss
            
    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters with weight decay.
    
        Args:
           v_0: activities or probabilities of visible layer (data to the RBM)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer (reconstruction)
           h_k: activities or probabilities of hidden layer (reconstruction)
        """
    
        # Compute the positive and negative phase associations
        pos_associations = np.dot(v_0.T, h_0) / self.batch_size
        neg_associations = np.dot(v_k.T, h_k) / self.batch_size
    
        # Calculate the gradients
        dW = pos_associations - neg_associations
    
        # Introduce weight decay (L2 regularization)
        weight_decay = self.weight_decay * self.weight_vh  # λ * W
        dW -= weight_decay  # Subtract weight decay from weight updates
    
        # Bias updates remain unchanged
        db_v = np.mean(v_0 - v_k, axis=0)
        db_h = np.mean(h_0 - h_k, axis=0)
    
        # Apply momentum and update parameters
        self.delta_weight_vh = self.momentum * self.delta_weight_vh + self.learning_rate * dW
        self.delta_bias_v = self.momentum * self.delta_bias_v + self.learning_rate * db_v
        self.delta_bias_h = self.momentum * self.delta_bias_h + self.learning_rate * db_h
    
        self.weight_vh += self.delta_weight_vh
        self.bias_v += self.delta_bias_v
        self.bias_h += self.delta_bias_h
    
        return


    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # Compute probabilities
        proba = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_vh))
        
        # Compute activations of hidden layer (samples from the distribution)
        h = sample_binary(proba)
        
        return proba, h


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
 
            # Compute total input for each unit
            total_input = self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T)
    
            # Split into data and label parts
            data_part = total_input[:, :-self.n_labels]
            label_part = total_input[:, -self.n_labels:]
    
            # Compute probabilities using sigmoid activation
            data_proba = sigmoid(data_part)
            label_proba = softmax(label_part)
    
            # Sample activations
            data_activations = (data_proba > np.random.rand(*data_proba.shape)).astype(int)
            label_activations = (label_proba > np.random.rand(*label_proba.shape)).astype(int)
    
            # Concatenate back into a normal visible layer
            proba = np.hstack((data_proba, label_proba))
            v = sample_binary(proba)
            
        else:
                        
            # Compute probabilities
            proba = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T))
            
            # Compute activations of hidden layer (samples from the distribution)
            v = sample_binary(proba)
            
        return proba, v


    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # Compute probabilities
        proba = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_v_to_h))
        
        # Compute activations of hidden layer (samples from the distribution)
        h = sample_binary(proba)
        
        return proba, h


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
      
            raise RuntimeError("This RBM is at the top of the DBN and should not have directed connections.")
            
        else:
                        
            # Compute probabilities
            proba = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_v_to_h.T))
            
            # Compute activations of hidden layer (samples from the distribution)
            v = sample_binary(proba)
            
        return proba, v  
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
      
