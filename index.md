# Neural Turing Machines

These are, without a doubt, my favorite advancement in neural network technology. The goal of this blog is to elucidate the machinations of the NTM and use a TensorFlow implementation to demonstrate some of the tasks that they've been trained to perform. Here's the stuff I'm going to talk about:

1. Architecture overview
2. The math driving the NTM
3. Implementing a NTM in TensorFlow (using the RNNCell class)
4. Tasks from the original paper
5. MANN's and oneshot learning

The NTM is demonstrably well-suited to performing memory-related tasks, such as: storing and recalling a sequence of bits, associative recall, and, to some extent, sorting data based on priority values (section 4). Additionally, further work has shown that vanilla NTM's are quite capable of oneshot learning. In oneshot learning, also known as meta-learning, is a method of teaching neural networks *how to learn* (section 5).

### General Workings

NTM's fall under the category of recurrent neural networks (RNN's). Traditional RNN's store a *representation* of the data that's been seen at previous timesteps. The NTM has the potential to store *all* data that from previous timesteps and perform operations explicitly on that data. This makes the NTM well-suited for learning small programs

The NTM consists of four core components (see figure ...):

* Controller network
* Read head
* Write head
* Memory matrix

![Basic NTM Diagram][ntm_basic_diagram]

[ntm_basic_diagram]: https://github.com/therealjtgill/therealjtgill.github.io/edit/master/assets/ntm_diagram_small.png

The controller's job is to learn how to produce activations that read from, and write to, the memory matrix according to the process specified by the training data. In essence, the controller attempts to learn a program that allows it to use the read and write heads as advantageously as possible.

The controller network can consist of any combination of feed-forward neural networks (FFNN's) and RNN's, but various experiments by the good people at Deep Mind have shown that an RNN controller produces the best results. The only restriction on the controller network is the number of outputs on the output layer; it produces the values that are used to read and write from the memory matrix.

Both the read and write heads consist of a **soft attention** mechanism that allow them to focus on parts of the memory matrix. 
