# Features

## Design

- Fine-grained loop-level primitives for tensor program transformations.
  - Arithmetic "completeness", i.e., support for basic arithmetic operations between loop iterators.
  - Canonicalization rules to reduce redundancy.
  - Automatic differentiation based on this representation.
  - Algorithm to partition the graph into subgraphs of a computation graph.
- Search. Key idea: matching the shape.
  - Bottom-up search and shape matching algorithm to avoid variables and restructure the design space into a tree.
  - Metric for finalizability (shape matching): shape distance, for pruning.

## Implementation

- Search.
  - MCTS with RAVE and progressive widening, and beam search.
  - Distributed and thread-safe framework for parallel search.
- Halide, PyTorch and TVM code generation.
