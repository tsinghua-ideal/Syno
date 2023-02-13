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

### Color Semantics

The dimensions of each tensor are assigned a unique color. It is possible for intermediate dimensions to have more than one color, or no color at all.

| Primitives | Forward | Backward |
| --- | --- | --- |
| `Share` | Two input dimensions shall not have common color. The output color is the union of input colors. | |
| `Stride` | The input and output dimensions shall not have color. | |
| `Unfold` | The window dimension is assigned no color. | |
| `Merge` | Blend the colors. | |
| `Split` | Duplicate the color. | |
