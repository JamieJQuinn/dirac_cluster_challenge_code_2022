# Initialize the data arrays
m = 2**11
n = 2**10
p = 2**9

A = np.full((m, n), 1, np.float) # matrix containing all 1's
B = np.full((n, p), 1, np.float) # matrix containing all 1's

# Copy the arrays to the device
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)

# Allocate memory on the device for the result
d_C= cuda.device_array((m, p))

@cuda.jit
def matmul(out, A, B):
    """Perform matrix multiplication of out = A * B
    """
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        out[i, j] = tmp

threadsperblock = (32, 32)

blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

total_time = 0.
for i in range(N_ITERATIONS):
    start = perf_counter()
    matmul[blockspergrid, threadsperblock](d_C, d_A, d_B)
    end = perf_counter()
    total_time += end-start
print(f"Mean GPU runtime: {total_time/N_ITERATIONS}")

# Copy the result back to the host
C = d_C.copy_to_host()

total_time = 0.
for i in range(N_ITERATIONS):
    start = perf_counter()
    C = A.dot(B)
    end = perf_counter()
    total_time += end-start
print(f"Mean GPU runtime: {total_time/N_ITERATIONS}")
