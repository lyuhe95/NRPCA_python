## Manifold Denoising by Nolinear Robust Principal Analysis (NRPCA)
This is the Python implementation of the NeurIPS paper “Manifold Denoising by Nolinear Robust Principal Analysis”. The paper is available at: [https://proceedings.neurips.cc/paper/2019/hash/a76c0abe2b7b1b79e70f0073f43c3b44-Abstract.html] A MATLAB implementation can be found at https://github.com/rrwng/NRPCA

## Overview

We extend robust principal component analysis to nolinear manifolds, where we assume that the data matrix contains a sparse component and a component drawn from some low dimensional manifold. We aim at separating both components from noisy data by proposing an optimization framework.

Compared with the original algorithm in the paper and the MATLAB implementation, here we use a simplified algorithm which avoids the estimation of curvatures. The modified loss function is $\min_S \sum_i ||C(\widetilde{X}^{(i)}-P_i(S))||_*+\beta||P_i(S)||_1$. We would like to thank Dr. Masayuki Aino for suggesting this simplification to us.

### Descriptions

**main.py**: Implements the algprithm NRPCA

**Example_MNIST.ipynb**: Code for MNIST digits 4&9 classification using NRPCA

**Example_SwissRoll.ipynb**: Code for 20 dimenssional SwissRoll dataset using NRPCA

