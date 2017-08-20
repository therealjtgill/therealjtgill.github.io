# **Neural Turing Machines**

These are, without a doubt, my favorite advancement in neural network technology. The goal of this blog is to elucidate the machinations of the NTM and use a TensorFlow implementation to demonstrate some of the tasks that they've been trained to perform. Here's the stuff I'm going to talk about:

1. Architecture overview
2. The math driving the NTM
3. Implementing a NTM in TensorFlow (using the RNNCell class)
4. Tasks from the original paper
5. MANN's and oneshot learning

The NTM is demonstrably well-suited to performing memory-related tasks, such as: storing and recalling a sequence of bits, associative recall, and, to some extent, sorting data based on priority values (section 4). Additionally, further work has shown that vanilla NTM's are quite capable of oneshot learning. In oneshot learning, also known as meta-learning, is a method of teaching neural networks *how to learn* (section 5).

## Architecture Overview

NTM's fall under the category of recurrent neural networks (RNN's). Traditional RNN's store a *representation* of the data that's been seen at previous timesteps. The NTM has the potential to store *all* data that from previous timesteps and perform operations explicitly on that data. This makes the NTM well-suited for learning small programs.

The NTM consists of four core components (see figure below). They all inter-rely on each other heavily, so it's hard to talk about one component without talking about the others... but this organization seems to make sense:

* Memory matrix
* Controller network
* Read/write heads

![Basic NTM Diagram](/assets/ntm_diagram_small.png)

### Memory Matrix

The memory matrix literally just saves information that the controller tells it to save. Data in the memory matrix is accessed by addressing a particular *row* of the matrix. An example memory matrix is shown below.

![Example memory matrix](/assets/mem.png)

The matrix is 15x8, meaning that there are 15 memory addresses (rows) that can have 8 "bits" of information stored in each address. I have to use "bits" in quotation marks because the values are in the range [0, 1]. Analog bits...?

### Controller Network

The controller's job is to learn how to produce activations that read from, and write to, the memory matrix according to the process specified by the training data. In essence, the controller attempts to learn a program that allows it to use the read and write heads as advantageously as possible.

The controller network can consist of any combination of feed-forward neural networks (FFNN's) and RNN's, but various experiments by the good people at Deep Mind have shown that an RNN controller produces the best results. The only restriction on the controller network is the number of outputs on the output layer; it produces the values that are used to read and write from the memory matrix.

![Controller](/assets/controller_small.png)

The boxes in green show the layers of the controller. The arrows entering/exiting the controller layers on the left/right sides show that the layers can possibly be RNN's. The vector values emitted from the last layer of the controller network are used to write-to and read-from memory. The nodes/neurons on the last controller layer are split into pieces, and several different activation functions are applied to those pieces (the different activation functions allow us to control the range of values we get from the controller).

### Read/Write Heads

Both the read and write heads consist of a **soft attention** mechanism that allow them to focus on parts of the memory matrix. What's cool is that the read/write heads can also choose to focus on *none* of the values in memory. When you imagine a soft attention mechanism imagine a one-hot vector, or any vector with normalized elements (see below).

![Attention mechanism example](/assets/attention%2Bmemory.png)

The row with the bright spot corresponds to the row of the memory matrix that we're giving attention to.

In addition to an attention mechanism, the write head produces **erase** and **add** vectors which remove data from memory and add data to memory, respectively. This can be used to overwrite or modify an existing entry, or add information to an unused memory location.
