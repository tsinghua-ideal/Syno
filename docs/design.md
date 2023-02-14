# Design

## Primitives

Fine grained primitives that work on loop iterators are devised. Along with the access semantics (in backward style), they are:
- `Share: [i, i] -> [i]`.
- `Stride: [i * s] -> [i]`.
- `Unfold: [i + j - k / 2] -> [i, j]`.
- `Merge: [i / n, i % n] -> i`.
- `Split: [i * n + j] -> [i, j]`.
- `MapReduce: [i] -> []`.
- `Shift: [(i + d) % n] -> [i]`.
- `Reverse: [n - 1 - i] -> [i]`.

### Shape Semantics

| Primitives | Forward | Backward |
| --- | --- | --- |
| `Share` | Zips two dimensions of equal sizes into one. | Duplicate a dimension. |
| `Stride` | Divides a dimension by a stride (coefficient). | Multiplies a dimension by a stride (coefficient). |
| `Unfold` | Creates a new dimension with size of kernel (coefficient). | Eliminates a dimension (all coefficients). |
| `Merge` | Merge. | Split. |
| `Split` | Split. | Merge. |
| `MapReduce` | Eliminates a dimension. | Creates a dimension of arbitrary size. |

`Shift` and `Reverse` have trivial shape semantics.

### Color Semantics

The dimensions of each tensor are assigned a unique color. It is possible for intermediate dimensions to have more than one color, or no color at all.

Note that `Share`-ing two dimensions with common color is catastrophic, as some entries in the tensor can no longer be accessed. Colors are devised to make the transforms legal, in the sense that all entries are utilized in the kernel.

Next, when we use the term dimension in the backward settings, we refer to the set of colors it has.

| Primitives | Forward | Backward |
| --- | --- | --- |
| `Share` | Two input dimensions shall not have common color. The output color is the union of input colors. | Add a new disjoint constraint, and substitute with the union of two dimensions. |
| `Stride` | The input and output dimensions shall not have color. | Simplify constraints with an empty set, and add it to known-clear-color list. |
| `Unfold` | The window dimension is assigned no color. | Remove from known-clear-color list, or simplify constraints with an empty set. |
| `Merge` | Blend the colors. | Substitute with the union of two dimensions. |
| `Split` | Duplicate the color. | Substitute the two output dimensions with the color of the new dimension and simplify. |

`MapReduce` does not have color semantics.

`Shift` and `Reverse` have trivial color semantics.
