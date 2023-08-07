# TODO List

## Core

- [ ] Do not distinguish between primary variables and coefficient variables.
- [ ] Fractional I/O shape.

### Autodiff

- [x] Fix propagation of iterator values by adding orientation to dimensions.
- [x] Add heuristics for choosing inner loop iterators in backward pipeline to enable loop fusions.

### Colors

- [x] Refine color semantics (determine more than 1 color when solving constraints).

### Hash

- [x] Fix hash collision. (If not possible, try finding data structures that tolerate collisions.) (Temporarily this seems fixed?)
- [x] Replace `std::hash` with custom hash for better reproducibility.
- [x] Use reproducible hash for `Node`.

## CodeGen

- [x] Generate object files, instead of static libraries. Then dynamically link the kernels instead of compiling C++ wrappers again and again.
- [x] Add auto scheduler options to reduce tuning time.
- [x] Add padding by padding variables that cannot be divided by their denominators.
- [ ] Optimization: Early reduction to reduce FLOPs.
- [x] Early reduction analysis is actually compulsory. Otherwise in autodiff settings, the RDom may be left unmentioned, causing Halide compile errors!
- [x] Sum -> Avg.
- [ ] Perform a trial to see if there are uncanonical case. (1 on weight)
- [x] Multiple weights. (Minimumize difference.)
- [x] Manual `rfactor` for up to 3x performance.
- [x] Add support for more arithmetic operations, other than product.
- [x] Add nested loops codegen for expression.
- [x] Generate metadata along with kernels.
- [x] Adjust the order of dimensions in weights for better cache locality.
- [ ] TVM codegen.
- [x] PyTorch codegen.
- [ ] Make padding algorithm primary-coefficient-ignorant.

## Transforms

- [x] Implement generation for ShiftOp.
- [ ] Add ReverseOp.
- [ ] Random shuffle convolution.
- [ ] What is a MergeOp by the way?
- [ ] To support Attention, what can we do?
- [ ] Make FinalizeOp a PrimitiveOp.
- [ ] Remove `priority` from `MapReduce`.
- [x] Add ExpandOp.

## Search

### Finalize

- [ ] Split Finalize, determine one weight at a time.
- [ ] Add canonicalization of chain of ShareOp.

### Generation

- [x] Before generating an Op, test sizes of new dimensions to make it legal.
- [x] Redesign generation algorithm for each Op.
- [x] Accept FLOPs constraints in Sampler, and generate MapReduceOp's accordingly.
- [ ] If ShapeComplexity finds the stage is in critical state, do not generate unnecessary Op's.
- [ ] Make `Allowance` depend on global invariant.

### Pruning

- [x] Add pruning with respect to finalizability criteria.
- [x] Add even more pruning with respect to finalizability criteria.
- [x] Canonicalize by pruning uncanonicalized Ops. (E.g., SplitOp and MergeOp should not be generated above Sum reductions, or below a weight dimension, e.t.c..)
- [x] Canonicalize transforms on weight. (E.g., SplitOp and MergeOp should not be generated below a weight dimension.)
- [x] `ShareOp::IsSharedDimensionCanonical()` still needs some modifications.
- [ ] Add mechanisms to automatically discover equivalent kernels. (TASO-like?)
- [ ] Unfolding some dimensions to output iterators seems to be not a good idea.
- [x] Make dead ends propagate. (By storing parent nodes.)
- [x] Canonicalization for reshape.

### Bindings

- [ ] Enable Explorer to visit a path.
- [ ] Make Explorer more command line friendly.
- [ ] Enable Explorer to print a tree.
- [x] Report dead ends to Python.
- [x] Add mocks for Node and Sampler.
- [ ] Standalone scheduler.
- [ ] 3-step searching.

### Misc

- [x] Index the search tree by hashes for reproducibility.
- [ ] Allow direct construction of kernel from primitives and parameters, without building the search tree. This can be done by serializing a DAG. Also save these data to files.
- [x] Multithreaded search tree building.
- [x] Compute a Path from a TensorView, to verify searchability.
- [x] Robustify the TensorView -> Path function.
- [ ] Add visualization for search space, e.g., draw the DAG interactively.
- [x] Node -> Graphviz.
- [x] Replace `Next` with `Arc`.

## Experiment

### MCTS

- [x] Multi-process support for MCTS. (Tree parallelization)
- [x] Multi-thread support for MCTS. (multi-thread simulation)
- [x] Two-step primitive selection. 
- [x] Implement FUSE

### More Scenarios

- [ ] Find more scenarios