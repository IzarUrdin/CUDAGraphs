# Burning GPU with CUDAGraphs

Probably the most significative advance in computing in this decade (the 2010´s) is quantum computing. However, CUDA programming and more specifically programming GPU with CUDAGraphs is the most reliable advance in computing.

Programming with CUDA_Graphs is like implementing a combinational circuit, similarly as programming with VHDL a Programmable Logic Device. It is a variant of dynamic parallel programming where each parallel thread can instantiate other parallel threads. Furthermore, each node of the graph can be saw as an atomic task but internally works with parallel threads.

![From https://devblogs.nvidia.com/cuda-graphs/](https://devblogs.nvidia.com/wp-content/uploads/2019/09/CUDA-Graphs.png)

Each node of the CUDAGraph can be another CUDAGraph or simply a Kernel function, which makes possible to manage the GPU to maximize its capabilities. Commonly a GPU has a clock frecuency of 1Ghz in contrast of the 3Ghz (at most) of a CPU, but a GPU has about than 340 of thread-cores against the 8 of a good desktop computer. As child nodes are executed in parallel we can scale-out horizontaly in each "level" of the graph up to 340 thread-cores so we can work 340/(3x8) = 14 faster than a very good PC. 

In this example, I want to show the implementation of an exploratory algorithm using CUDAGraphs with the framework of ManagedCUDA for C#, which can be downloaded from the NuGet or from this [repository](https://github.com/kunzmi/managedCuda). I expect to explore all the posibilites of a mutual-exclusion matrix using CUDAGraph technology to reach a performance of Log(nVariables^2/nThreads).

**The Mutex Problem**

The Mutual-Exclusion Algorithm (Mutex) is used in very wide engineering and applied sciences proceedings to obtain different logics from data. It can be used, for example, in helping to do the most effective treatment for a patient. If we know that a patient has had his appendix removed then he is not suffering an appendicitis attack. Defining a matrix with all the mutual-exclusions variables of diseases we can approach diagnosis and explore the most effective treatment. However, even discarding the 80% of the variables, we will have dozens of unanswered variables so we have to explore all the combinations to get the most effective sequence to reach a diagnose.

In the image below we can see a mutex-matrix with 7 decision variables.

![Mutex-Matrix](https://ixilka.net/publications/mutex-matrix.jpg)

Due to the mutual-exclusion of the variable *c* with *b,f* and *g*, answering variable *c*, for example, "no blood in urine", we discard urethra (*g*), bladder (*b*) or kidney (*f*) infection. So next step is to ask for *a* or *d*. But if we cannot determine if the blood in urine is significant or not, we must explore the other ways.

Going through the most effective way is to get first the next most determinant and effective answer and we don't know which is the next if we don't explore all the variants until end. So knowing the weight (cost-risk-effective-probability) of each variable we can decide the best way on proceeding. That´s the basis of the experts systems, CAD and a lot of Machine Learning systems and sometimes resolving that may cost years of computing for a single CPU.

**The final goal**

Deciding which is the best procedure in each step is like calculating the best move in a chess game, but having a huge number of variables and results of the different proceedings it is possible to make-up a regressive exploration to determine different groups of exploratory models as Mutex-Matrix reductions. Hence, even when computers are not diagnosing particular cases they are mining data from historical procedings to define better models of diagnosing.

# FIRST ATTEMPT

Something that can be useful for beginners with CUDA is that launching a child kernel passing a local variable it is not allowed. You will see the exception: `Error: a pointer to local memory cannot be passed to a launch as an argument`.

It is possible to solve copying with `memalloc` for using it as a referable argument.

Here is a simplified kernel as an example of deep recursion.

    __device__ char init[6] = { "12345" };
    
    __global__ void Recursive(int depth, const char* route) {
    	
    	// up to depth 6
    	if (depth == 5) return;
    	
    	//declaration for a referable argument (point 6)
    	char* newroute = (char*)malloc(6); 
    	memcpy(newroute, route, 5);
    			
    	int o = 0;
    	int newlen = 0;
    	for (int i = 0; i < (6 - depth); ++i)
    	{
    		if (i != threadIdx.x)
    		{
    			newroute[i - o] = route[i];
    			newlen++;
    		}
    		else
    		{
    			o = 1;
    		}
    	}
    	
    	printf("%s\n", newroute);
    
    	Recursive <<<1, newlen>>>(depth + 1, newroute);
    
    }
    
    __global__ void RecursiveCount() {
    	Recursive <<<1, 5>>>(0, init);
    }
    
**Optimus Mutex**
    
Basing the algorithm in the recursive kernel above we can explore all the possible combinations having into account the mutual exclusion of the variables shown in the matrix.

[Algorithm will be showed soon]
    
Counting all the recursive calls (k-operations) for this matrix we obtain the number 3554. In the worst case the number of k-opearions will be the factorial of the number of variables; in this example of not optimized 7 variables it could results in 7x6x5x4x3x2x1 = 5040 k-operations. Due to the mutual exclusion configuration of the variables, the number of k-operations may be reduced drastically. When a mutex is optimized we use only the variables that are entangled enough to be a knowledge on a problem; others only represent noise that worths nothing in a solution. 
    
![Mutex-Matrix-10](https://ixilka.net/publications/mutex10_1.jpg)

This optimized mutex for 10 variables needs only 3.539 k-operations to be resolved instead of the 10! = 3.628.800 operations for a 10 variable mutex not optimized. 

To identify the optimum mutex is another problem, for now we only need to know that they can exist.
