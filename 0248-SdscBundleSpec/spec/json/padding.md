# Padding

For window and padded operations such as convolution, padding information is
carried per dimension in `paddingSizes_`, a field of
[DataStructDims](datastructdims.md). Because `DataStructDims` appears both as the
total-size dimensions (`N_`) and inside each entry of
[`dataStageParam_`](datastageparam.md), padding must be set in **both** places for
a padded dimension.

## `paddingSizes_`

`paddingSizes_` is keyed by the name of the padded dimension. One entry is required
for each dimension that is padded.

```
"paddingSizes_": {
  "<padded dim>": {
    "padFront_":         <integer>,
    "padBack_":          <integer>,
    "unneededPad_":      <integer>,
    "unneededPadFront_": <integer>,
    "unneededPadBack_":  <integer>,
    "totalSize_":        <integer | "N/A">,
    "stride_":           <integer>,
    "dilation_":         <integer>,
    "windowDim_":        "<window dim name>"
  }
}
```

### Where `paddingSizes_` appears

`paddingSizes_` is keyed by spatial dimension label (`"i"`, `"j"`) and appears in
three places in a document, because `DataStructDims` itself appears in three places:

| Location | Sizes used |
|---|---|
| `N_` — the operation's total dimensions | the full-output padding sizes |
| `dataStageParam_["0"].ss_` — per-core steady state | the per-core sizes when the operation has them, otherwise the full-output sizes |
| `dataStageParam_["0"].el_` — per-core epilogue | the same as `ss_` |

A spatial dimension is **omitted entirely** from `paddingSizes_` when its kernel
window dimension (`ki_`, `kj_`) has been squeezed out of the iteration space — that
is, when the kernel size in that dimension is 1.

Operations that are not windowed — pointwise, matmul, reductions, transpose — carry
no `paddingSizes_` at all.

### Fields

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `padFront_` | integer | No | Default `0` | Front padding, in elements. Set to `-1` when the padded dimension is chunked across cores — see below. |
| `padBack_` | integer | No | Default `0` | Back padding, in elements. Set to `-1` under the same condition as `padFront_`. |
| `unneededPad_` | integer | No | — | Total padding elements that are not needed by the computation. |
| `unneededPadFront_` | integer | No | — | The portion of `unneededPad_` originating from `padFront_`. |
| `unneededPadBack_` | integer | No | — | The portion of `unneededPad_` originating from `padBack_`. |
| `totalSize_` | integer or string | No | The string `"N/A"` only when `windowDim_` is unset and `padFront_` is negative | Total size of the dimension including padding. |
| `stride_` | integer | No | **>= 1**; default `1` | Stride applied by the operation. A stride of `0` would not advance. |
| `dilation_` | integer | No | **>= 1**; default `1` | Dilation applied by the operation. |
| `windowDim_` | string | No | A primary dimension name, e.g. `"kij_"` | The window dimension associated with this padded dimension. |

### `totalSize_` versus the dimension size

`totalSize_` is the input size **after** padding, whether or not padding is
applied. The dimension's own size — in `N_` or in a `dataStageParam_` entry — is the
**output** size.

For a 128×128 image with a 3-wide window:

| | `padFront_`/`padBack_` = 1 | no padding |
|---|---|---|
| `totalSize_` | 130 | 128 |
| dimension size | 128 | 126 |

### Dimensions chunked across cores

When a padded dimension is divided across cores, set `padFront_` and `padBack_` to
`-1` in the `"core"` datastage. This marks the padding as distributed rather than
belonging to any single core's tile.

### Which fields appear, by operation

The `unneededPad*` fields are emitted only where the operation can absorb padding
that the computation does not need.

| Operation | Fields emitted | Where the sizes come from |
|---|---|---|
| **avgpool2d** | The base set only — `padFront_`, `padBack_`, `totalSize_`, `stride_`, `dilation_`, `windowDim_` — for each surviving spatial dimension (`i`, `j`). No `unneededPad*`. | One set of sizes, used for both `ss_` and `el_`. |
| **depthwise conv2d** | The base set **plus** `unneededPad_`, `unneededPadFront_` and `unneededPadBack_`, for each surviving spatial dimension. | Two variants: the full output size for `N_`, and a per-core output slice for `ss_` and `el_`. |

For avgpool2d, `totalSize_` is `(out - 1) * stride + kernel`, `dilation_` is `1`, and
`windowDim_` is the corresponding kernel dimension (`ki_` for `i`, `kj_` for `j`).

For depthwise conv2d, `padFront_` and `padBack_` are the effective pads remaining
after the unneeded portion is subtracted; `unneededPad_` is the total padding beyond
the minimum required input span, and `unneededPadFront_`/`unneededPadBack_` divide it
between the two ends.

## Padding state per dimension

The padding *state* of a dimension — as opposed to its sizes — is the `padding`
field on [CoordinateInfo](coordinateinfo.md), which takes one of six string values
(`"nopad"`, `"lowered_padded"`, `"padded_nozeropad"`, `"padded_wzeropad"`,
`"padded_fullspan"`, `"padded_fullspan_wunneeded"`).

Two of those values are worth calling out for convolution:

| Value | Used for |
|---|---|
| `"padded_nozeropad"` | `conv2d` when padding is non-zero |
| `"padded_fullspan_wunneeded"` | `conv2d` when padding is zero |

Note that `padding_` on [ScheduleTreeNode](scheduletreenode.md) is a different
field: it is an **object** holding padding parameters for an allocation, not one of
the values above.

---

[↑ SuperDSC-Bundle specification](../../SuperDSC-Bundle.md)
