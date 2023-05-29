# TODO List

## Core

### Autodiff

- [x] Fix propagation of iterator values by adding orientation to dimensions.
- [ ] Add heuristics for choosing inner loop iterators in backward pipeline to enable loop fusions.

### Colors

- [x] Refine color semantics (determine more than 1 color when solving constraints).

### Hash

- [x] Fix hash collision. (If not possible, try finding data structures that tolerate collisions.) (Temporarily this seems fixed?)
- [ ] Replace `std::hash` with custom hash for better reproducibility.

## CodeGen

- [x] Generate object files, instead of static libraries. Then dynamically link the kernels instead of compiling C++ wrappers again and again.
- [ ] Add auto scheduler options to reduce tuning time.
- [x] Add padding by padding variables that cannot be divided by their denominators.
- [ ] Optimization: Early reduction to reduce FLOPs.
- [ ] Early reduction analysis is actually compulsory. Otherwise in autodiff settings, the RDom may be left unmentioned, causing Halide compile errors!

## Transforms

- [ ] Add ReverseOp.

## Search

### Generation

- [x] Before generating an Op, test sizes of new dimensions to make it legal.
- [x] Redesign generation algorithm for each Op.
- [ ] Accept FLOPs constraints in Sampler, and generate MapReduceOp's accordingly.

### Pruning

- [x] Add pruning with respect to finalizability criteria.
- [ ] Add even more pruning with respect to finalizability criteria.
- [x] Canonicalize by pruning uncanonicalized Ops. (E.g., SplitOp and MergeOp should not be generated above Sum reductions, or below a weight dimension, e.t.c..)
- [ ] Canonicalize transforms on weight. (E.g., SplitOp and MergeOp should not be generated below a weight dimension.)
- [ ] Add mechanisms to automatically discover equivalent kernels. (TASO-like?)

### Misc

- [x] Index the search tree by hashes for reproducibility.
- [ ] Allow direct construction of kernel from primitives and parameters, without building the search tree.
- [ ] Discover full-dead-end subtrees and report to Python.
- [ ] Multithreaded search tree building.

## Experiment

- [x] Build framework for MNIST experiments. 
- [x] Multi-process support for MCTS. (Tree parallelization)
    - [x] server
    - [x] client
    - [x] Tree parallelization
    - [x] setup script
- [x] Two-step primitive selection. 
- [ ] Early Stopping during training. (Not urgent yet. )
- [-] Search Result Saving (path, mcts, etc.). 
- [x] Cache searched nodes. 
- [ ] Add MNIST MLP models. 
- [ ] Work out conv search. 
- [ ] Reward formatting. 
- [ ] Parametrize filtering and reward returning. 