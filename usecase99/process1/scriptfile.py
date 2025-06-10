import cupy as cp
import sys


if __name__ == "__main__":
    
    with open(sys.argv[1], "r") as f:

        n = int(f.read().strip())
    
    print(n)
    
    matrix_0 = cp.random.rand(n,n)
    matrix_1 = cp.random.rand(n,n)

    matrix_out = cp.dot(matrix_0, matrix_1)

