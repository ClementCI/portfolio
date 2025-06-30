# Portfolio



Welcome to my portfolio of machine learning and data science projects, developed as part of my academic coursework. Each project explores a different area of modern AI, from fundamental algorithms to deep probabilistic models and neural networks. All implementations are original and emphasize conceptual clarity, practical experimentation, and reproducibility. Among these, the [Embedded Topic Model for Document Analysis](#embedded-topic-model-for-document-analysis) and [CNN Architectures for Image Classification: From VGG to ConvNeXt](#cnn-architectures-for-image-classification-from-vgg-to-convnext) stand out as key projects, showcasing up-to-date techniques in natural language processing and computer vision.







## Table of Contents



### Fundamental Machine Learning Projects

- [Decision Tree Learning on the MONK Datasets](#decision-tree-learning-on-the-monk-datasets)

- [Support Vector Machine Classifier](#support-vector-machine-classifier)

- [Bayesian Classification and Boosting](#bayesian-classification-and-boosting)

- [Supervised Classification Challenge](#supervised-classification-challenge)



### Probabilistic Machine Learning Projects

- [Coordinate Ascent Variational Inference](#coordinate-ascent-variational-inference)

- [Stochastic Variational Inference for Latent Dirichlet Allocation](#stochastic-variational-inference-for-latent-dirichlet-allocation)

- [Reparameterization of Common Distributions](#reparameterization-of-common-distributions)

- [Variational Autoencoder (VAE) for MNIST Latent Representation and Image Generation](#variational-autoencoder-vae-for-mnist-latent-representation-and-image-generation)

- [Embedded Topic Model for Document Analysis](#embedded-topic-model-for-document-analysis)



### Vision Projects

- [Filtering Operations](#filtering-operations)

- [Edge Detection and Hough Transform](#edge-detection-and-hough-transform)

- [Image Matching & 3D Reconstruction](#image-matching--3d-reconstruction)



### Deep Learning Projects

- [Multi-Layer Perceptrons for Classification and Function Approximation](#multi-layer-perceptrons-for-classification-and-function-approximation)

- [Hopfield Networks for Associative Memory](#hopfield-networks-for-associative-memory)

- [Deep Belief Networks with Restricted Boltzmann Machines](#deep-belief-networks-with-restricted-boltzmann-machines)

- [Image Classification with a 1-layer Network](#image-classification-with-a-1-layer-network)

- [Image Classification with a 2-layer Network](#image-classification-with-a-2-layer-network)

- [Image Classification with a Convolutional Neural Network](#image-classification-with-a-convolutional-neural-network)

- [CNN Architectures for Image Classification: From VGG to ConvNeXt](#cnn-architectures-for-image-classification-from-vgg-to-convnext)

- [Character-Level Language Modeling with RNN](#character-level-language-modeling-with-rnn)



### Dimensionality Reduction Project

- [Dimensionality Reduction of MEP Voting Data](#dimensionality-reduction-project)







## Fundamental Machine Learning Projects





The `fundamental_ml` folder showcases basic machine learning concepts through carefully implemented classification algorithms and structured experiments. Each project demonstrates a key method in supervised learning, with a focus on both theoretical grounding and practical evaluation. All implementations are built from scratch to enhance understanding of core ML techniques such as probabilistic modeling, tree-based learning, ensemble methods, and support vector machines.





---

### Decision Tree Learning on the MONK Datasets



This project implements decision tree learning from scratch using a greedy top-down approach, applied to the MONK-1, MONK-2, and MONK-3 datasets. These synthetic datasets are designed to test concepts such as entropy, information gain, overfitting, and the effects of pruning. The project includes training unpruned and pruned trees, evaluating their performance, and visualizing the learned models. Key results show how pruning can improve generalization, particularly for MONK-1 and MONK-3, while MONK-2 remains challenging due to its irregular class structure.



[Go to Decision Tree Learning on the MONK Datasets](./fundamental_ml/decision_trees/README.md)





### Support Vector Machine Classifier



This project builds a Support Vector Machine (SVM) classifier from scratch using the dual optimization formulation and kernel methods. It supports linear, polynomial, and radial basis function (RBF) kernels and visualizes how each affects the decision boundary. The implementation includes margin tuning via the regularization parameter ($C$), with experiments conducted on synthetic 2D datasets to explore separability, overfitting, and kernel expressiveness.



[Go to Support Vector Machine Classifier](./fundamental_ml/support_vector_machines/README.md)





### Bayesian Classification and Boosting



This project implements and evaluates a Naive Bayes classifier and its boosted variant using AdaBoost, comparing their performance to decision trees across several datasets (Iris, Vowel, and Olivetti Faces). It explores how boosting improves classification accuracy, especially for complex datasets, and includes PCA for dimensionality reduction and insightful decision boundary visualizations.



[Go to Bayesian Classification and Boosting](./fundamental_ml/bayes_classifier_and_boosting/README.md)









### Supervised Classification Challenge



This project tackles a supervised classification problem involving a labeled dataset and a separate evaluation set with hidden labels. Multiple models were tested, including K-Nearest Neighbors, SVMs, ensemble methods (Random Forest, AdaBoost, XGBoost), and a 2-layer feedforward neural network. Comprehensive preprocessing steps—such as imputation, encoding, scaling, feature selection, and PCA—were applied. The neural network achieved the highest validation accuracy (70.6%) and was used to generate the final predictions for submission.



[Go to Supervised Classification Challenge](./fundamental_ml/classification_challenge/README.md)









## Probabilistic Machine Learning Projects



The `probabilistic_ml` folder explores key ideas in Bayesian machine learning and variational inference through a series of carefully designed projects. Each implementation focuses on modeling uncertainty, approximating complex posteriors, or enabling scalable inference in probabilistic frameworks. From latent variable models and topic modeling to reparameterization techniques and variational autoencoders, these projects emphasize both mathematical rigor and empirical validation, with custom implementations built from scratch or in PyTorch.



---





### Coordinate Ascent Variational Inference



This project implements Coordinate Ascent Variational Inference (CAVI) to approximate the posterior distribution in a Bayesian model with Normal-Gamma priors over Gaussian observations. By deriving and applying manual update rules, the method is tested on synthetic datasets of increasing size. The results show that as sample size grows, the variational posterior increasingly matches the exact posterior, confirmed via ELBO convergence and contour plots comparing all estimators.



[Go to Coordinate Ascent Variational Inference](./probabilistic_ml/coordinate_ascent_variational_inference/README.md)





### Stochastic Variational Inference for Latent Dirichlet Allocation



This project implements Stochastic Variational Inference (SVI) for Latent Dirichlet Allocation (LDA), following Hoffman et al. (2013). It compares SVI with traditional Coordinate Ascent Variational Inference (CAVI) on synthetic datasets of increasing size to evaluate convergence speed and scalability. Both methods are benchmarked using ELBO tracking and runtime analysis, showing that while CAVI achieves marginally better ELBO, SVI dramatically improves computational efficiency and handles large-scale data effectively.



[Go to Stochastic Variational Inference for Latent Dirichlet Allocation](./probabilistic_ml/stochastic_variational_inference_lda/README.md)





### Reparameterization of Common Distributions



This project focuses on enabling gradient-based optimization for models involving the Beta  and Dirichlet distributions using differentiable reparameterization techniques. By leveraging the Kumaraswamy distribution for Beta and a softmax-Gaussian approximation for Dirichlet, the implementation provides efficient samplers that support backpropagation through stochastic layers—crucial in variational inference frameworks such as VAEs. Visual comparisons validate the accuracy of the reparameterized methods against traditional samplers.





[Go to Reparameterization of Common Distributions](./probabilistic_ml/reparameterization/README.md)





### Variational Autoencoder (VAE) for MNIST Latent Representation and Image Generation



A deep generative model is implemented to learn compressed latent representations of handwritten digits. The project compares Kullback-Leibler divergence and Maximum Mean Discrepancy as regularizers in the VAE objective. It visualizes the structure of the learned latent space and generates new digits from noise. Results show how regularization impacts the expressiveness and geometry of latent embeddings.



[Go to VAE for MNIST Latent Representation and Image Generation](./probabilistic_ml/variational_autoencoders/README.md)





### Embedded Topic Model for Document Analysis



This project implements the Embedded Topic Model (ETM), which fuses traditional topic modeling with neural word embeddings by embedding both topics and words into the same semantic space. Built in PyTorch and evaluated on the 20 Newsgroups dataset, the model learns to represent documents as mixtures of embedded topics using variational inference. The project compares ETM against Latent Dirichlet Allocation (LDA), evaluating coherence, diversity, and perplexity across multiple vocabulary sizes. Results confirm that ETM offers superior interpretability and richer semantic structure, though it requires careful tuning to match LDA’s predictive reliability.



[Go to Embedded Topic Model for Document Analysis](./probabilistic_ml/embedded_topic_model/README.md)







## Vision Projects



The `vision` folder contains projects focused on classical computer vision techniques, emphasizing image analysis through filtering, edge detection, geometric modeling, and 3D reconstruction. Each project demonstrates foundational methods used in low- and mid-level vision, combining theoretical insight with practical experimentation. From frequency-domain processing and multiscale edge detection to feature-based matching and geometric transformations, the implementations offer a comprehensive view of early visual pipelines.



---



### Filtering Operations



This project explores the use of the Discrete Fourier Transform (DFT) for image processing, with a focus on frequency-based filtering. Through hands-on experiments, it investigates how different frequencies contribute to image content and how filters like Gaussian and median impact noise reduction and visual clarity. The study highlights the role of phase and magnitude in the Fourier domain, the visual artifacts caused by aliasing, and the benefits of smoothing prior to subsampling. Comparisons between filtering methods show tradeoffs between noise removal and preservation of image details.



[Go to Filtering Operations](./vision/filtering_operations/README.md)



### Edge Detection and Hough Transform



This project explores edge detection through multiscale differential operators and line extraction via the Hough transform. Edges are detected using Gaussian-derivative filters capturing first and second-order changes, followed by zero-crossing detection and gradient-based thresholding. The project compares a standard Hough transform with a gradient-weighted variant that improves robustness by aligning votes with edge orientations. Results show how careful tuning of parameters and using gradient direction leads to more precise line detection.



[Go to Edge Detection and Hough Transform](./vision/edge_detection_and_hough_transform/README.md)





### Image Matching & 3D Reconstruction



This project explores how to recover scene geometry from images using feature-based methods. It includes robust estimation of homographies and fundamental matrices through SIFT keypoints and RANSAC. Depending on whether the scene is planar or has depth variation, either a homography or a fundamental matrix is used to align images or reconstruct 3D point clouds via triangulation. Experiments show the sensitivity of these models to noise and feature distribution, and how techniques like RANSAC and synthetic control of focal lengths can enhance geometric interpretation.



[Go to Image Matching & 3D Reconstruction](./vision/image_matching_and_3d_reconstruction/README.md)





## Deep Learning Projects



The `deep_learning` folder features a collection of end-to-end deep learning projects built from scratch or using PyTorch. Covering a wide range of architectures and tasks—from feedforward neural networks to convolutional and recurrent models—these projects emphasize implementation, training, and evaluation of deep networks. Key topics include gradient validation, unsupervised pretraining, image classification, generative models, architectural ablations, and sequence modeling. Each project explores how network design, regularization, and optimization influence generalization and performance on real-world datasets.





---



### Multi-Layer Perceptrons for Classification and Function Approximation



This project implements and investigates multi-layer perceptrons (MLPs) for supervised tasks such as binary classification and continuous function approximation. Using a NumPy-based MLP with one hidden layer, the study evaluates how architectural and training parameters affect generalization, convergence, and overfitting. Key experiments explore the impact of hidden layer size, online vs. batch learning, and learning rate adaptation. Results demonstrate that smaller models can still perform well on simple tasks, while deeper architectures require careful regularization on noisy or imbalanced datasets.



[Go to Multi-Layer Perceptrons for Classification and Function Approximation](./deep_learning/classification_regression_mlp/README.md)



### Hopfield Networks for Associative Memory



This project explores the capacity and retrieval dynamics of Hopfield networks trained using Hebbian learning. By encoding binary patterns and recalling them from distorted versions, the network demonstrates properties such as energy minimization, convergence to stable attractors, and sensitivity to sparsity and correlation. The implementation compares synchronous and asynchronous updates, investigates the effects of pattern overlap and self-connections, and confirms theoretical memory capacity bounds under orthogonal patterns.



[Go to Hopfield Networks for Associative Memory](./deep_learning/hopfield_net_associative_memory/README.md)





### Deep Belief Networks with Restricted Boltzmann Machines



This project implements a Deep Belief Network (DBN) composed of stacked Restricted Boltzmann Machines (RBMs), trained layer-by-layer using Contrastive Divergence. Built entirely in NumPy, the network learns hierarchical representations from binary MNIST data. After unsupervised pretraining, the model is evaluated through reconstruction quality and classification performance using a shallow classifier on the final hidden layer. The DBN reaches a test accuracy of 88.21% without supervised fine-tuning, and demonstrates generative capabilities via Gibbs sampling.



[Go to Deep Belief Networks with Restricted Boltzmann Machines](./deep_learning/deep_belief_net_rbm/README.md)







### Image Classification with a 1-layer Network



This project implements a fully connected one-layer neural network from scratch for image classification on CIFAR-10. It focuses on validating analytical gradient computations and studying the impact of various training strategies and hyperparameters. The model is trained using mini-batch gradient descent with two loss functions: categorical cross-entropy and multiple binary cross-entropy. The experiments explore the effects of learning rate, regularization, loss choice, and data augmentation, demonstrating reliable gradient accuracy and good generalization under optimized settings, reaching over 42% accuracy.



[Go to Image Classification with a 1-layer Network](./deep_learning/image_classification_one_layer_net/README.md)







### Image Classification with a 2-Layer Network



This project focuses on training a fully connected 2-layer neural network from scratch for image classification on the CIFAR-10 dataset. It emphasizes analytical gradient validation, hyperparameter tuning, and optimization strategies. The model is trained using mini-batch gradient descent, with experiments comparing cyclical learning rate schedules and the Adam optimizer. Results show that carefully tuned regularization, wider architectures, and data augmentation lead to improved generalization, with the best model achieving over 57% test accuracy.



[Go to Image Classification with a 2-layer Network](./deep_learning/image_classification_two_layer_net/README.md)





### Image Classification with a Convolutional Neural Network



This project implements and trains a convolutional neural network entirely with NumPy to classify images from the CIFAR-10 dataset. The goal is to understand low-level operations in CNNs, including convolution, pooling, and backpropagation, without using any deep learning frameworks. The model features manually coded forward and backward passes, cyclical learning rate schedules, label smoothing, and offline data augmentation. To speed up computations, convolutions are implemented efficiently using matrix multiplication (im2col-style reshaping). The model reaches a test accuracy of 69.13% and includes thorough benchmarking against PyTorch implementations.



[Go to Image Classification with a Convolutional Neural Network](./deep_learning/image_classification_ConvNet/README.md)



### CNN Architectures for Image Classification: From VGG to ConvNeXt



This project explores the evolution of convolutional neural network architectures for image classification by training and evaluating models from scratch on CIFAR-10, CIFAR-100, and ImageNette datasets. It starts with simple VGG-style models and progressively integrates architectural improvements such as batch normalization, global average pooling, attention modules (SE and CBAM), and robust loss functions. The project also benchmarks ConvNeXt-Tiny on ImageNette and studies training under noisy labels using Symmetric Cross-Entropy. Extensive ablation studies and visualizations accompany each experiment to assess the contribution of regularization, architecture, and optimization strategies.



[Go to CNN Architectures for Image Classification: From VGG to ConvNeXt](./deep_learning/image_classification_advanced_ConvNet/README.md)





### Character-Level Language Modeling with RNN



This project involves building a recurrent neural network (RNN) from scratch using NumPy to perform character-level language modeling. The model is trained on a cleaned version of *"Harry Potter and the Goblet of Fire"*, aiming to predict the next character in a sequence and generate plausible text. The implementation includes manual forward and backward passes, gradient validation against PyTorch, and use of the Adam optimizer. Experiments explore the effects of sequence randomization, batch size, and sampling strategies (temperature scaling and nucleus sampling) on model performance and text generation quality.



[Go to Character-Level Language Modeling with RNN](./deep_learning/language_modeling_rnn/README.md)







## Dimensionality Reduction Project



The `dimensionality_reduction` project explores the ideological structure of the European Parliament by applying dimensionality reduction to voting records of Members of the European Parliament (MEPs). A custom similarity function, robust to abstentions and missing values, is used to measure alignment between MEPs. Classical Multidimensional Scaling (MDS) is then applied to project these relationships into two dimensions. Temporal dynamics are also explored by computing time-based MDS projections and aligning them with Procrustes analysis to visualize ideological shifts across four legislative periods.



[Go to Dimensionality Reduction Project](./dimensionality_reduction/README.md)



