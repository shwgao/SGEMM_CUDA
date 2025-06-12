
## Warp
- The threads of a block are grouped into so-called warps, consisting of **32 threads**.
- A warp is then assigned to a warp scheduler, which is the **physical core** that executes the instructions.
- There are **four** warp schedulers per multiprocessor.
- **Global memory coalescing**. It’s the most important thing to keep in mind when optimizing a kernel’s GMEM memory accesses toward achieving the peak bandwidth.

## Grid and Block

<div align="center">
<img src="./assets/CUDA_thread_hierarchy.png" alt="Grid and Block" width="800"/>
</div>

syntax for grid and block:
```c++
dim3 grid(blockIdx.x, blockIdx.y, blockIdx.z);
dim3 block(threadIdx.x, threadIdx.y, threadIdx.z);
```
launch kernel:
```c++
kernel<<<grid, block>>>(args);
```

In reality, `blockIdx.x` controls the row id of the block, `blockIdx.y` controls the column id of the block. When you launch the kernel to compute the matrix with dimension of `M*N`, you may set the `blockIdx.x` to `(M+BLOCKSIZE-1)/BLOCKSIZE` and `blockIdx.y` to `(N+BLOCKSIZE-1)/BLOCKSIZE`.

So in your code, when you want to get the row id of the block, you can use `blockIdx.x * BLOCKSIZE` and the column id of the block, you can use `blockIdx.y * BLOCKSIZE`.
In this case, the `blockIdx.x` and `blockIdx.y` in the figure seems a little bit confusing.


Of course, you can change the meaning of `blockIdx.x` and `blockIdx.y` to control the column id and row id of the block respectively. But you need to take care of your code when you do this.


## Global Memory Coalescing
In reality, the GPU supports 32B, 64B and 128B memory accesses. So, if each thread is loading a 32bit float from global memory, the warp scheduler (probably the MIO) can coalesce this 32*4B=128B load into a single transaction. 

<div align="center">
<img src="./assets/coalescing.png" alt="Global Memory Coalescing" width="600"/>
</div>

To implement this, you only need to change the thread mapping strategy.

```c++
const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

**Improvement**: `300GFLOPS` (Naive) --> `2000GFLOPS` (Coalesced).

## Shared Memory Cache-Blocking

Looking at the hierarchy of GPU memory:

<div align="center">
<img src="./assets/memory-arch.png" alt="Memory Hierarchy" width="600"/>
</div>

- Threads can communicate with the other threads in its block via the shared memory. 
- The maximum of shared memory for one block can up to 168 KB based on different compute capabilities.
- Bandwidth of shared memory is about 12TB/s, compared to 750GB/s for global memory. For Volta.








## Debugging
1. ncu can't be used on the school's machines.
2. `make: *** No rule to make target 'build'.  Stop.`: I ran `make` in the wrong directory.

## References 

- https://siboehm.com/articles/22/CUDA-MMM  (Highly recommended)