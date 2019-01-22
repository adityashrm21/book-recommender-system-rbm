# Book-Recommender-System-RBM
A book recommender system created using Restricted Boltzmann Machines in Python. For a detailed tutorial on how to build this system, read my blog post on [Building a Recommender System using Restricted Boltzmann Machines](https://adityashrm21.github.io/Book-Recommender-System-RBM/).

<img src = "https://img.shields.io/badge/requirements-compatible-blue.svg">

- Python 3.6 and above
- CUDA 8.0
- Tensorflow-gpu 1.4.1 (older - compatible with CUDA 8.0)
- NumPy
- Pandas
- Matplotlib

<img src = "https://cdn-images-1.medium.com/max/1600/1*cLusB9Kkfaf-EudwtX8ASQ.png">

[Image Source](https://cdn-images-1.medium.com/max/1600/1*cLusB9Kkfaf-EudwtX8ASQ.png)

## How to use this repository?

The repository contains a single script that can be run to produce the results. You will need to have the required dataset to run the code locally. You can either:
- Clone the repository and use the script and the files in the [data folder](https://github.com/adityashrm21/Book-Recommender-System-RBM/tree/master/data).
- If you want to contribute, fork the repository and clone it and create pull requests with the additions/changes.
- Simply download the code file and get the whole dataset including a couple more files from the [goodbooks-10  dataset repository](https://github.com/zygmuntz/goodbooks-10k) if you want to play around a bit more.


<img src = "https://media.giphy.com/media/3o7abuqxszgO6pFb3i/giphy.gif">

[GIF Source](https://media.giphy.com/media/3o7abuqxszgO6pFb3i/giphy.gif)

## What are Restricted Boltzmann Machines?

**Note**: A detailed explanation on RBMs can be found on my [blog post](https://adityashrm21.github.io/Restricted-Boltzmann-Machines/).

RBMs are two-layered artificial neural network with generative capabilities. They have the ability to learn a probability distribution over its set of input. RBMs were invented by Geoffrey Hinton and can be used for dimensionality reduction, classification, regression, collaborative filtering, feature learning and topic modeling.

RBMs are a special class of [Boltzmann Machines](https://en.wikipedia.org/wiki/Boltzmann_machine) and they are restricted in terms of the connections between the visible and the hidden units. As stated earlier, they are a two-layered neural network (one being the visible layer and the other one being the hidden layer) and these two layers are connected by a fully bipartite graph. This means that every node in the visible layer is connected to every node in the hidden layer but no two nodes in the same group are connected to each other. This restriction allows for more efficient training algorithms than are available for the general class of Boltzmann machines, in particular the [gradient-based](https://en.wikipedia.org/wiki/Gradient_descent) contrastive divergence algorithm.


A Boltzmann Machine             |  A Restricted Boltzmann Machine
:-------------------------:|:-------------------------:
<img src = "https://upload.wikimedia.org/wikipedia/commons/7/7a/Boltzmannexamplev1.png" width = "300">  |  <img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Restricted_Boltzmann_machine.svg/440px-Restricted_Boltzmann_machine.svg.png" width = "300">


Each undirected edge in a Boltzmann Machine represents a dependency. In this example there are 3 hidden units and 4 visible units. In an RBM, we have a symmetric bipartite graph where no two units within a same group are connected.
Multiple RBMs can also be `stacked` and can be fine-tuned through the process of gradient descent and back-propagation. Such a network is called as a Deep Belief Network. Although RBMs are occasionally used, most people in the deep-learning community have started replacing their use with General Adversarial Networks or Variational Autoencoders.

RBM is a Stochastic Neural Network which means that each neuron will have some random behavior when activated. There are two other layers of bias units (hidden bias and visible bias) in an RBM. This is what makes RBMs different from autoencoders. The hidden bias RBM produce the activation on the forward pass and the visible bias helps RBM to reconstruct the input during a backward pass.

<img src = "https://image.slidesharecdn.com/mlss2014xamatriain-140721124307-phpapp02/95/recommender-systems-machine-learning-summer-school-2014-cmu-72-638.jpg?cb=1405946863">

[Image Source](https://image.slidesharecdn.com/mlss2014xamatriain-140721124307-phpapp02/95/recommender-systems-machine-learning-summer-school-2014-cmu-72-638.jpg?cb=1405946863)

## Why use RBMs for recommendation?

RBMs are unsupervised learning algorithms which try to **reconstruct** user input and in order to achieve this, they try to learn patterns from the examples in our data. This is then used to create a lower-dimensional representation of the pattern which can later be used to reconstruct approximations of the original input. Their ability to do this makes them a good fit for our problem because we need the algorithm to identify a pattern (the reading taste of a user) from the input and reconstruct it in the form of a score for each book (a rating essentially). This ultimately would help us in providing recommendations to that user based on the reconstructed scores.

## Dataset

The dataset was obtained from the **goodbooks-10k** dataset from [this github repository](https://github.com/zygmuntz/goodbooks-10k). This data is also available at other places like Kaggle.
## Architecture

We are implementing and using RBMs to train our network for the task. The network was trained for 25 epochs with a mini-batch size of 50. The code is using tensorflow-gpu version 1.4.1 which is compatible with CUDA 8.0 (you need to use compatible versions of tensorflow-gpu and CUDA). You can check the version of TensorFlow compatible with the CUDA version installed on your machine [here](https://www.tensorflow.org/install/source#tested_source_configurations). You can also use the CPU-only version of TensorFlow if don't have access to a GPU or if you are okay with the code running for a little more time.

### Training Error Graph

This graph was obtained on training the data for 60 epochs with a mini-batch size of 100:

![error](imgs/error60.png)

## Deciding number of Epochs

<center><img src ="imgs/free_energy.png"></center>

The plot shows the average free energy for training and the validation dataset with epochs.

If the model is not overfitting at all, the average free energy should be about the same on training and validation data. As the model starts to overfit the average free energy of the validation data will rise relative to the average free energy of the training data and this gap represents the amount of overfitting. So we can determine the number of epochs to run the training for using this approach. Looking at the plot, we can safely decide the number of epochs to be around **25**.

## Contribute

#### To-do tasks:
- Create a web application (maybe integrate into TensorFlow.js) for people to use the it online where they can just select the books they have read to get recommendations.
- Clean up the code and make it more object oriented.

**Feel free to add any ideas by opening issues or contributing in any way you like!**

#### Sources/References:
* [Wikipedia - Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
*  https://github.com/srp98/Movie-Recommender-using-RBM
* [Guide on training RBM by Geoffrey Hinton](https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf)
* https://skymind.ai/wiki/restricted-boltzmann-machine
* https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
* [Artem Oppermann's Medium post on understanding and training RBMs]( https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-ii-4b159dce1ffb)
