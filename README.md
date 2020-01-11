# Burning GPU with CUDAGraphs

Probably the most significative advance in computing in this decade (the 2010´s) is quantum computing. However, CUDA programming and more specifically programming GPU with CUDAGraphs is the most reliable advance in computing.

Programming with CUDA_Graphs is like implementing a combinational circuit, similarly as programming with VHDL a Programmable Logic Device. It is a variant of dynamic parallel programming where each parallel thread can instantiate other parallel threads. Furthermore, each node of the graph can be saw as an atomic task but internally works with parallel threads.

![From https://devblogs.nvidia.com/cuda-graphs/](https://devblogs.nvidia.com/wp-content/uploads/2019/09/CUDA-Graphs.png)

Each node of the CUDAGraph can be another CUDAGraph or simply a Kernel function, which makes possible to manage the GPU to maximize its capabilities.

In this example, I want to show the implementation of an exploratory algorithm using CUDAGraphs with the framework of ManagedCUDA for C#, which can be downloaded from the NuGet or from this [repository](https://github.com/kunzmi/managedCuda)

