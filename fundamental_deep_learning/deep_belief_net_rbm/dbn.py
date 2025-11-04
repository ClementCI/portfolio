from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                             <---> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size, period=1000),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size, period=1000),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size, period=1000)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 400
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        #vis = (vis > 0.5).astype(np.float32) # Convert to binary as when we trained

        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # Drive the network bottom to top
        h_hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[0] # Hidden layer
        h_pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(h_hid)[0] # Penultimate layer
        
        # Run alternating Gibbs sampling in the top RBM
        v_top = np.hstack((h_pen, lbl)) # First visible computation by concatenating with labels
        
        for _ in range(self.n_gibbs_recog):
            h_top = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_top)[0]  # Update hidden probabilities
            v_top = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_top)[0] # Update visible probabilities
        
        lbl = sample_categorical(v_top[:, -true_lbl.shape[1]:]) # Extract labels by sampling
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return 100.*np.mean(np.argmax(lbl,axis=1)==np.argmax(true_lbl,axis=1))

    def generate(self,true_lbl,name, idx=0):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # Initialize the top layer with input labels and random hidden activations
        v_top = np.hstack((sample_binary(np.random.rand(n_sample, self.sizes['hid'])), lbl))

        # Run Gibbs sampling in the top RBM
        for _ in range(self.n_gibbs_gener):
            h_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(v_top)[-1] # Update hidden activations
            v_top[:,: self.sizes['hid']] = self.rbm_stack['pen+lbl--top'].get_v_given_h(h_top)[-1][:,: self.sizes['hid']] # Update non-label activations

            v_sample = v_top[:,: self.sizes['hid']] # Extract binary values for the non-label part
            
            # Drive the network top to bottom
            v_hid = self.rbm_stack["hid--pen"].get_v_given_h_dir(v_sample)[-1]  # Hidden layer
            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(v_hid)[0]  # Bottom visible layer as probabilities
            
            # Record a frame
            # For a single sample (n_sample=1), just reshape that one. If you have multiple, you might tile them or pick one to display. 
            frame_data = vis[0].reshape(self.image_size)  # pick the first in batch
            im_artist = ax.imshow(frame_data, cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)
            records.append([im_artist])
        
        
        anim = stitch_video(fig, records).save("%s.generate%d.gif" % (name, np.argmax(true_lbl)))
        plt.close(fig)
        
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations, period=1000):
        """
        Greedy layer-wise training by stacking RBMs. Trains each layer independently using Contrastive Divergence.
    
        Args:
          vis_trainset: Training images (size: number of samples x visible layer size)
          lbl_trainset: Training labels (size: number of samples x label layer size)
          vis_testset: Test images (size: number of samples x visible layer size)
          lbl_testset: Test labels (size: number of samples x label layer size)
          n_iterations: Number of training iterations (mini-batch based)
        """
        
        losses = []
        names = ["vis--hid", "hid--pen", "pen+lbl--top"]
        
        try:
            self.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
    
            self.loadfromfile_rbm(loc="trained_rbm", name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
    
            self.loadfromfile_rbm(loc="trained_rbm", name="pen+lbl--top")        
    
        except IOError:
            print("Training vis--hid")
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")
            losses.append(self.rbm_stack["vis--hid"].get_reconstruction_loss())
    
            print("Training hid--pen")
            hidden_representation = self.rbm_stack["vis--hid"].get_h_given_v(vis_trainset)[-1]
            self.rbm_stack["hid--pen"].cd1(hidden_representation, n_iterations)
            self.rbm_stack["vis--hid"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")
            losses.append(self.rbm_stack["hid--pen"].get_reconstruction_loss())
            
            print("Training pen+lbl--top")
            pen_representation = self.rbm_stack["hid--pen"].get_h_given_v(hidden_representation)[-1]
            top_trainset = np.hstack((pen_representation, lbl_trainset))
            self.rbm_stack["pen+lbl--top"].cd1(top_trainset, n_iterations)
            self.rbm_stack["hid--pen"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")
            losses.append(self.rbm_stack["pen+lbl--top"].get_reconstruction_loss())
            
        # Plot all losses after training
        for i in range(len(losses)):
            iterations = range(0, n_iterations, period)
            plt.plot(iterations, losses[i], label=f"{names[i]}")

        plt.title("Reconstruction Loss Over Iterations for Different RBMs")
        plt.xlabel("Iterations")
        plt.ylabel("Average Reconstruction Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"loss_comparison_rbms.png")
        plt.close()
            

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
