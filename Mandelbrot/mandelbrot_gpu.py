from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def mandelbrot_kernel(data, xlow, xhigh, ylow, yhigh):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    def mapFromTo(x, a, b, c, d):
        y = (x - a) / (b - a) * (d - c) + c
        return y

    x = mapFromTo(tx, 0, row, xlow, xhigh)
    y = mapFromTo(ty, 0, col, ylow, yhigh)
    c = complex(x , y)
    z = 0.0j
    for i in range(50):
        z = z ** 2 + c
        if (z.real ** 2 + z.imag ** 2) >= 4:
            data[tx, ty] = i
            break
        else:
            data[tx, ty] = 100
    
row = 1000
col = 1000
plot = np.zeros([row, col])
mandelbrot_kernel[row, col](plot, -2, 1, -1, 1)

fig = plt.figure(dpi=200)
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(plot.T, cmap="RdBu", interpolation="bilinear")

plt.show()