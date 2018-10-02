# Book-Recommender-System-RBM
A book recommender system created using simple Restricted Boltzmann Machines

## What are Restricted Boltzmann Machines?

RBMs are a two-layered artificial neural network with generative capabilities. They have the ability to learn a probability distribution over its set of input. RBMs were invented by Geoffrey Hinton and can be used for dimensionality reduction, classification, regression, collaborative filtering, feature learning and topic modeling.

RBMs are a special class of [Boltzmann Machines](https://en.wikipedia.org/wiki/Boltzmann_machine) and they are restricted in terms of the connections between the visible and the hidden units. As stated earlier, they are a two-layered neural network (one being the visible layer and the other one being the hidden layer) and these two layers are connected by a fully bipartite graph. This means that every node in the visible layer is connected to every node in the hidden layer but no two nodes in the same group are connected to each other. This restriction allows for more efficient training algorithms than are available for the general class of Boltzmann machines, in particular the [gradient-based](https://en.wikipedia.org/wiki/Gradient_descent) contrastive divergence algorithm.


A Boltzmann Machine             |  A Restricted Boltzmann Machine
:-------------------------:|:-------------------------:
<img src = "https://upload.wikimedia.org/wikipedia/commons/7/7a/Boltzmannexamplev1.png" width = "300">  |  <img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Restricted_Boltzmann_machine.svg/440px-Restricted_Boltzmann_machine.svg.png" width = "300">


Each undirected edge in a Boltzmann Machine represents a dependency. In this example there are 3 hidden units and 4 visible units. In an RBM, we have a symmetric bipartite graph where no two units within a same group are connected.
Multiple RBMs can also be `stacked` and can be fine-tuned through the process of gradient descent and back-propagation. Such a network is called as a Deep Belief Network. Although RBMs are occasionally used, most people in the deep-learning community have started replacing their use with General Adversarial Networks or Variational Autoencoders.
