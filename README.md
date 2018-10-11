# Book-Recommender-System-RBM
A book recommender system created using Restricted Boltzmann Machines in Python

<img src = "https://cdn-images-1.medium.com/max/1600/1*cLusB9Kkfaf-EudwtX8ASQ.png">

[Image Source](https://cdn-images-1.medium.com/max/1600/1*cLusB9Kkfaf-EudwtX8ASQ.png)

<img src = "https://i.gifer.com/Td5I.gif">

[GIF Source](https://i.gifer.com/Td5I.gif)

If you are feeling like that, read on to understand what are RBMs and why are we using them for this task!

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

## Why use RBM for recommendation?

RBMs are unsupervised learning algorithms which try to **reconstruct** user input and in order to achieve this, they try to learn patterns from the examples in our data. This is then used to create a lower-dimensional representation of the pattern which can later be used to reconstruct approximations of the original input. Their ability to do this makes them a good fit for our problem because we need the algorithm to identify a pattern (the reading taste of a user) from the input and reconstruct it in the form of a score for each book (a rating essentially). This ultimately would help us in providing recommendations to that user based on the reconstructed scores.

## Requirements:

- Python 3.6 and above
- Tensorflow 1.4.1 (can be newer if a different CUDA version is installed)
- CUDA 8.0 (Optional - if you have access to a GPU)
- NumPy
- Pandas
- Matplotlib

## Dataset

The dataset was obtained from the **goodbooks-10k** dataset from [this github repository](https://github.com/zygmuntz/goodbooks-10k). This data is also available at other places like Kaggle.
## Architecture

We are implementing and using RBMs to train our network for the task. The network was trained for 25 epochs with a mini-batch size of 50. The code is using tensorflow-gpu version 1.4.1 which is compatible with CUDA 8.0 (you need to use compatible versions of tensorflow-gpu and CUDA). You can check the version of TensorFlow compatible with the CUDA version installed on your machine [here](https://www.tensorflow.org/install/source#tested_source_configurations). You can also use the CPU-only version of TensorFlow if don't have access to a GPU or if you are okay with the code running for a little more time.

#### Sources/References:
* [Wikipedia - Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
*  https://github.com/srp98/Movie-Recommender-using-RBM
* [Guide on training RBM by Geoffrey Hinton](https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf)
* https://skymind.ai/wiki/restricted-boltzmann-machine
* https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
* [Artem Oppermann's Medium post on understanding and training RBMs]( https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-ii-4b159dce1ffb)
