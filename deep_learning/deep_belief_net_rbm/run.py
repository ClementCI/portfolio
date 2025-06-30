from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

# for the entire training set: n_train=60000

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")
    
    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=10,
                                     
    )
    
    rbm.cd1(visible_trainset=train_imgs, n_iterations=10000)
    
    
    print ("\nStarting a comparison between Restricted Boltzmann Machines..")

    hidden_sizes = [500, 400, 300, 200]
    n_iterations = 10000
    period = 100
    loss_histories = {}
    
    for h in hidden_sizes: 
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                         ndim_hidden=h,
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=10,
                                         period=period)
        
        rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations)
        loss_histories[h] = rbm.get_reconstruction_loss()  # Store loss for this RBM

    # Plot all losses after training
    for h, loss in loss_histories.items():
        iterations = range(0, n_iterations, period)
        plt.plot(iterations, loss, label=f"Hidden Units: {h}")

    plt.title("Reconstruction Loss Over Iterations for Different Hidden Sizes")
    plt.xlabel("Iterations")
    plt.ylabel("Average Reconstruction Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"loss_comparison_hidden_sizes.png")
    plt.close()
       
    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000, period=1000)
    
    
    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)
    
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms", idx=digit)
        
