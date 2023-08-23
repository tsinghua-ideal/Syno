# Design

## Primitives

Fine grained primitives that work on loop iterators are devised. Along with the access semantics (in backward style), they are:
- `Share: [i, i] -> [i]`.
- `Stride: [i * s] -> [i]`.
- `Unfold: [i + j - k / 2] -> [i, j]`.
- `Merge: [i / n, i % n] -> [i]`.
- `Split: [i * n + j] -> [i, j]`.
- `Reduce: [i] -> []`.
- `Shift: [(i + d) % n] -> [i]`.
- `Reverse: [n - 1 - i] -> [i]`.

We believe that these primitives constitute a big enough search space for tensor program kernels.

### Shape Semantics

| Primitives | Forward | Backward |
| --- | --- | --- |
| `Share` | Zips two dimensions of equal sizes into one. | Duplicate a dimension. |
| `Stride` | Divides a dimension by a stride. | Multiplies a dimension by a stride. |
| `Unfold` | Creates a new dimension with size of kernel. | Eliminates a dimension. |
| `Merge` | Merge. | Split. |
| `Split` | Split. | Merge. |
| `Reduce` | Eliminates a dimension. | Creates a dimension of arbitrary size. |

`Shift` and `Reverse` have trivial shape semantics.

### Pruning Formalism

TODO
