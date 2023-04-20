# TODO List

## Core

### Autodiff

- [ ] Fix propagation of iterator values by adding orientation to dimensions.
- [ ] Add heuristics for choosing inner loop iterators in backward pipeline to enable loop fusions.

### Colors

- [ ] Refine color semantics (determine more than 1 color when solving constraints).

### Hash

- [ ] Fix hash collision. (If not possible, try finding data structures that tolerate collisions.)

## CodeGen

- [ ] Add padding by padding variables that cannot be divided by their denominators.

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

- [ ] Build framework for experiments and consider which models to use.
