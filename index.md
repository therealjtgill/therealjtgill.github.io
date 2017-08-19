# Neural Turing Machines

These are, without a doubt, my favorite advancement in neural network technology. The goal of this blog is to elucidate the machinations of the NTM and use a TensorFlow implementation to demonstrate some of the tasks that they've been trained to perform. Here's the stuff I'm going to talk about:

1. Architecture overview
2. The math driving the NTM
3. Implementing a NTM in TensorFlow (using the RNNCell class)
4. Tasks from the original paper
5. MANN's and oneshot learning

### General Workings

The NTM is modeled after the Turing Machine, and consists of four core components (see figure ...):

* Controller network
* Read head
* Write head
* Memory matrix

![Basic NTM Diagram][ntm_basic_diagram]

[ntm_basic_diagram]: /assets/ntm_diagram_small.png
