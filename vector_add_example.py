import numpy as np
from numba import cuda
from time import perf_counter

print(cuda.gpus)

data = np.ones(256*4096*16) # ~16 million
data_gpu = np.zeros_like(data)

threadsperblock = 256

blockspergrid = (data.size + (threadsperblock-1))//threadsperblock

print(f"Threads per block: {threadsperblock}")
print(f"Blocks per grid: {blockspergrid}")
print(f"Total threads: {blockspergrid*threadsperblock}")
print(f"data.size: {data.size}")

@cuda.jit
def times2(out, in):
    pos = cuda.grid(1) # blockIdx.x * blockDim.x + threadIdx.x
    if pos < in.size:
        out[pos] *= in[pos]*2

N_ITERATIONS = 10

total_time = 0.
for i in range(N_ITERATIONS):
    start = perf_counter()
    times2[blockspergrid, threadsperblock](data_gpu, data)
    end = perf_counter()
    total_time += end-start
print(f"Mean GPU runtime: {total_time/N_ITERATIONS}")

total_time = 0.
for i in range(N_ITERATIONS):
    start = perf_counter()
    data_cpu = data*2
    end = perf_counter()
    total_time += end-start
print(f"Mean CPU runtime: {total_time/N_ITERATIONS}")

