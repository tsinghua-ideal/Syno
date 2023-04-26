# TODO List

## Core

### Autodiff

- [x] Fix propagation of iterator values by adding orientation to dimensions.
- [ ] Add heuristics for choosing inner loop iterators in backward pipeline to enable loop fusions.

### Colors

- [ ] Refine color semantics (determine more than 1 color when solving constraints).

### Hash

- [x] Fix hash collision. (If not possible, try finding data structures that tolerate collisions.) (Temporarily this seems fixed?)

## CodeGen

- [ ] Add padding by padding variables that cannot be divided by their denominators.
- [ ] Optimization: Early reduction to reduce FLOPs.
- [ ] Early reduction analysis is actually compulsory. Otherwise in autodiff settings, the RDom may be left unmentioned, causing Halide compile errors!

## Transforms

- [ ] Add ReverseOp.

## Search

### Generation

- [ ] Before generating an Op, test sizes of new dimensions to make it legal.
- [ ] Redesign generation algorithm for each Op.

### Pruning

- [ ] Add pruning with respect to finalizability criteria.
- [ ] Canonicalize by pruning uncanonicalized Ops. (E.g., SplitOp and MergeOp should not be generated above Sum reductions, or below a weight dimension, e.t.c..)
- [ ] Add mechanisms to automatically discover equivalent kernels. (TASO-like?)

### Misc

- [ ] Index the search tree by hashes for reproducibility.

## Experiment

- [x] Build framework for MNIST experiments. 
- [] Multi-processing support for MCTS. 
- [] Two-step primitive selection. 