# **Neural Turing Machines**

These are, without a doubt, my favorite advancement in neural network technology. The goal of this blog is to elucidate the machinations of the NTM and use a TensorFlow implementation to demonstrate some of the tasks that they've been trained to perform. Here's the stuff I'm going to talk about:

1. Architecture
2. Math
3. Implementation in TensorFlow
4. Training tasks
5. Oneshot learning

The NTM is demonstrably well-suited to performing memory-related tasks, such as: storing and recalling a sequence of bits, associative recall, and, to some extent, sorting data based on priority values (section 4). Additionally, further work has shown that vanilla NTM's are quite capable of oneshot learning. Oneshot learning, also known as meta-learning, is a method of teaching neural networks *how to learn* (section 5).

## Architecture

NTM's fall under the category of recurrent neural networks (RNN's). Traditional RNN's store a *representation* of the data that's been seen at previous timesteps. The NTM has the potential to store *all* data that from previous timesteps and perform operations explicitly on that data. This makes the NTM well-suited for learning small programs. The NTM works well with time-series data, and in most applications the NTM is presented with *sequences* of information, and then asked to perform some sort of operation (or operations) on that data, and that operation usually ends up being time-dependent.

The NTM consists of four core components (see figure below). They all inter-rely on each other heavily, so it's hard to talk about one component without talking about the others... but this organization seems to make sense:

* External memory matrix
* Controller network
* Read/write heads

![Basic NTM Diagram](/assets/ntm_diagram_small.png)

### External Memory Matrix

The memory matrix is literally just a repository of information that the controller saves. Data in the memory matrix is accessed (for reading or writing) by addressing a particular *row* of the matrix. The matrix is called 'external' because it isn't trained through error backpropagation. An example memory matrix is shown below.

![Example memory matrix](/assets/mem.png)

The matrix is 15x8, meaning that there are 15 memory addresses (rows) that can have 8 "bits" of information stored in each address. I have to use "bits" in quotation marks because the values are in the range [0, 1]. Analog bits...?

### Controller Network

The controller's job is to learn how to produce activations that read-from and write-to the memory matrix according to the process specified by the training data. In essence, the controller attempts to learn a program that allows it to use the read and write heads as advantageously as possible.

The controller network can consist of any combination of feed-forward neural networks (FFNN's) and RNN's, but various experiments by the good people at Deep Mind have shown that an RNN controller produces the best results. The only restriction on the controller network is the number of outputs on the output layer; for every timestep in the task, the controller produces values that are used to read and write from the memory matrix.

![Controller](/assets/controller_small.png)

The boxes in green show the layers of the controller. The arrows entering/exiting the controller layers on the left/right sides show that the layers can possibly be RNN's. The vector values emitted from the last layer of the controller network are used to write-to and read-from memory. **Most importantly**, the nodes/neurons on the last controller layer are split into pieces, and several different activation functions are applied to those pieces. The different activation functions allow us to control the range of values we get from the controller, and ultimately let us perform the **reading and writing operations**.

### Read/Write Heads

Both the read and write heads consist of a **soft attention** mechanism that allow them to focus on parts of the memory matrix. What's cool is that the read/write heads can also choose to focus on *none* of the values in memory. When you imagine a soft attention mechanism imagine a one-hot vector, or any vector with normalized elements (see below).

![Attention mechanism example](/assets/attention%2Bmemory.png)

The row with the bright spot corresponds to the row of the memory matrix that we're giving attention to.

In addition to an attention mechanism, the write head produces **erase** and **add** vectors which remove data from memory and add data to memory, respectively. This can be used to overwrite or modify an existing entry, or add information to an unused memory location. An example of this is shown below (white pixels have values close to 1, black pixels close to 0).

![Writing to memory example](/assets/write_memory_small.png)

The read and write heads can produce attention at different places in memory, which allows the NTM to write-to memory and read-from memory separately. In this implementation we force the NTM to write first and read second.

## Math

All right, so where do the attention mechanisms come from? We said before that the read/write heads produce addresses can focus on different locations, but the math to create those addresses is exactly the same for both heads. The read/write heads produce a **k**<sub>t</sub>, **s**<sub>t</sub>, β<sub>t</sub>, g<sub>t</sub>, and γ<sub>t</sub> (see below). Note that emboldened variables represent vectors, capitalized variables are matrices, and lower-case, non-bolded variables are scalars.

![Last layer of controller](/assets/controller_out_small.png)

| Piece | Name        | Description                                                                           | Activation             |
|-------|-------------|---------------------------------------------------------------------------------------|------------------------|
| **k**<sub>t</sub> | Key         | Emitted for content-based addressing.                                                 | None                   |
| β<sub>t</sub>     | Attenuation | Sharpens address generated from content-based addressing.                             | Softplus               |
| g<sub>t</sub>     | Gate        | Interpolates between content-based address and address used at the previous timestep. | Sigmoid                |
| **s**<sub>t</sub> | Shift       | Shifts the current address to a potentially different location.                       | Softmax                |
| γ<sub>t</sub>     | Sharpen     | Sharpens the result of the shift operation.                                           | Oneplus (softplus + 1) |
| **e**<sub>t</sub> | Erase       | Removes data from the memory matrix.                                                  | Sigmoid                |
| **a**<sub>t</sub> | Add         | Adds data to the memory matrix.                                                       | Sigmoid                |

The first five variables are used to build addresses to access data from memory, and the last two are used to modify what's stored in memory. Each variable has a time-subscript, meaning that the controller can emit brand new values for these variables at each timestep in the task. For now we're only concerned with focusing on a particular memory address, which the NTM does by:

1. generating an address based on data already in the matrix (content-based addressing)
2. interpolating between the content-based address and the address used in the previous timestep
3. shifting the address to a new location using circular convolution
4. sharpening the result of the shifting operation.

