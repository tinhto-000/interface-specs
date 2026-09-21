# SuperDSC-Bundle Interface Specification

## Summary

This document specifies the `SuperDSC-Bundle`, the interface between the torch-spyre frontend compiler and the Spyre backend compiler (Deeptools).

## Scope

A SuperDSC-Bundle is the unit the frontend hands to the backend for one complex
kernel. This specification defines two things:

- the **SDSC JSON document format** — one document per Spyre operation, describing
  the operation, its tensors, and how its work is divided across cores;
- the **`sdscbundle` MLIR dialect** — the operations that sequence those documents,
  allocate device memory, and bind runtime values.

It defines what a producer may write and what a consumer accepts. It does not
specify how either compiler is implemented, how work should be divided for
performance, or how to author MLIR beyond the constraints stated here.

**Audience.** A reader implementing a producer of bundles, or a consumer of them.
Familiarity with MLIR and with the Spyre data-parallel execution model is assumed;
[`guide/`](guide/) is the introductory material and forms no part of this
specification.

**Conformance** is defined in [Conformance](spec/json/conformance.md): a conforming
document validates against [`sdscbundle-schema.json`](spec/sdscbundle-schema.json),
and satisfies the constraints stated on the object pages that the schema cannot
express.

## Normative references

| Reference | Role |
|---|---|
| [`spec/sdscbundle-schema.json`](spec/sdscbundle-schema.json) | The machine-readable contract for the JSON document format. Carries its own `version`. |
| [JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12) | The dialect the schema is written in. |
| [MLIR SCF](https://mlir.llvm.org/docs/Dialects/SCFDialect/), [Arith](https://mlir.llvm.org/docs/Dialects/ArithDialect/), [Affine](https://mlir.llvm.org/docs/Dialects/Affine/) dialects | Upstream operations permitted in a bundle, with the restrictions in [Permitted upstream MLIR operations](spec/dialect/MLIR-bundle-API.md#permitted-upstream-mlir-operations). |
| [torch-spyre](https://torch-spyre.readthedocs.io/) | The frontend that produces bundles, and the list of torch operations it supports. |

## Terms

| Term | Meaning |
|---|---|
| **SuperDSC** | The top-level object of an SDSC JSON document, describing one Spyre operation and its mapping across cores. |
| **DSC** (DesignSpaceConfig) | One work division of a SuperDSC. A SuperDSC carries an array of them; with balanced work division a single entry suffices. |
| **core** | A RaPiD core: one compute engine plus its LX scratchpad. Cores are indexed from `0`. |
| **corelet** | A subdivision within a core. The corelet fold factor is `2`. |
| **fold** | One level of subdivision of a dimension, with a `factor_` giving the number of slices at that level. Folds nest, and their factors multiply to the dimension's element count. |
| **work division** | The assignment of an operation's dimension slices to cores, expressed by `numWkSlicesPerDim_` and `coreIdToWkSlice_` on the SuperDSC. |
| **stick** | The innermost contiguous group of elements that is transferred and operated on as a unit. Its dimensions and sizes are given by `stickDimOrder_` and `stickSize_` in a `primaryDsInfo_` entry. |
| **datastage** | A description of tile dimensions at one level of the hierarchy — a `dataStageParam_` entry, holding steady-state (`ss_`) and epilogue (`el_`) dimensions. |
| **symbolic value** | A value not fixed at compile time — either a tensor start address or a dimension size — resolved before the job is launched. |

## Execution model

`SuperDSC-Bundle` views Spyre at the data-parallel level of hardware abstraction:
multiple cores, each with a compute engine and a scratchpad memory, interfaced with
each other and with off-chip memory banks over an on-chip interconnect fabric.

<p align="center">
  <img src="data_parallel_hw_abstraction.png" alt="data_parallel_hw_abstraction" width="450"/>
</p>
<p align="center">
  Figure 1. Hardware abstraction of the multi-core accelerator embodied in `SuperDSC-Bundle`
</p>

Against that abstraction, a bundle expresses:

* complex kernels comprising a sequence of operations;
* the work division — the computation split — across RaPiD cores for each operation;
* the placement of each operation's input and output tensors, in DDR memory or in the
  LX scratchpad of a core;
* operation and tensor shapes, either static or symbolic;
* tensor start addresses in DDR and LX, either fixed or symbolic.

The backend consumes a bundle to produce `SpyreCode`: the job binary, a job plan and
other compiled artifacts. Where a start address or a shape is symbolic, `SpyreCode`
carries variables to be substituted before execution, together with the mechanism
that applies those substitutions just before the job is launched.

### Related interfaces

* `SuperDSC`, without bundle capability, is the interface between the Deeptools
  frontend and backend compilers.
* This interface is expected to be superseded by Kernel Tile Intermediate
  Representation (KTIR); see the
  [KTIR specification](https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md).
* `SpyreCode` is tracked in
  [torch-spyre#277](https://github.com/torch-spyre/torch-spyre/issues/277).

## Structure of SuperDSC-Bundle

The backend expects the frontend to produce multiple output files, that work in conjuction to instruct the backend on how one or multiple programs can be compiled and executed in a sequence as part of a complex kernel. The expectation is to receive:
* one or more sdsc.json files, each describing a Spyre operation
* one mlir file with the SuperDsc-Bundle IR

### Data format naming

Data formats are named by their enumerated values throughout this specification —
`SEN169_FP16`, `IEEE_FP32`, `SEN143_FP8`, `SENINT8` and so on. These are the values
a `dataFormat_` field accepts; the full set is listed on
[LabeledDataStructure](spec/json/labeleddatastructure.md#data-formats) and in the
schema. Informal short forms are not used, so that a name in a table is always a
value a producer can write.

`SEN169_FP16` is DLFloat16, described in
[DLFloat: A 16-b Floating Point Format Designed for Deep Learning Training and
Inference](https://research.ibm.com/publications/dlfloat-a-16-b-floating-point-format-designed-for-deep-learning-training-and-inference).
The conversion operations spell it `DL16` inside their own identifiers —
`DL16TOFP32`, `FP32TODL16`, `FP8TODL16`, and their lowercase `opFuncName` strings.
Those are operation names, spelled exactly as shown, and refer to the same format.

### Specification contents

The normative contract for both file kinds is in [`spec/`](spec/) — the JSON
document format under [`spec/json/`](spec/json/) and the MLIR dialect under
[`spec/dialect/`](spec/dialect/). This document is the entry point; the pages below
are the specification.

**Start here**

| Page | What it covers |
|---|---|
| [Conformance](spec/json/conformance.md) | What makes a document conforming, what validation rejects, versioning |
| [JSON Schema](spec/json/json-schema.md) | The machine-readable contract, [`sdscbundle-schema.json`](spec/sdscbundle-schema.json) |
| [SDSC JSON API](spec/json/SDSC-json-api.md) | Overall file structure |
| [Object Hierarchy](spec/json/JSON-object-Hierarchy.md) | How the objects nest |

**The `sdscbundle` MLIR dialect**

| Page | What it covers |
|---|---|
| [MLIR Bundle API](spec/dialect/MLIR-bundle-API.md) | Bundle container, `sdsc_execute`, `device_mem_allocate`, symbolic values, and the upstream MLIR operations permitted in a bundle |

**JSON objects, by role**

| Role | Pages |
|---|---|
| Top level | [SuperDsc](spec/json/superdsc-object.md), [DesignSpaceConfig](spec/json/designspaceconfig.md) |
| Tensors | [LabeledDataStructure](spec/json/labeleddatastructure.md), [MemoryOrganization](spec/json/memoryorganization.md), [PrimaryDsInfo](spec/json/primarydsinfo.md), [DataStructDims](spec/json/datastructdims.md) |
| Folding | [FoldProperty](spec/json/foldproperty.md), [FoldManager](spec/json/foldmanager.md) |
| Layout | [Padding](spec/json/padding.md), [Stick Layout Constraints](spec/json/stick-layout-constraints.md) |
| Execution | [DataStageParam](spec/json/datastageparam.md), [ScheduleTreeNode](spec/json/scheduletreenode.md), [CoordinateContainer](spec/json/coordinatecontainer.md), [CoordinateInfo](spec/json/coordinateinfo.md) |
| Operations | [ComputeOperation](spec/json/computeoperation.md), [ConstantInfo](spec/json/constantinfo.md) |
| | [Reference documents](spec/reference.md) |

Non-normative introductory material, worked examples and tuning guidance are in
[`guide/`](guide/) and form no part of the contract.

### SuperDSC-Bundle intermediate representation in mlir

This intermediate representation (IR) in mlir conveys a complex kernel made of one or multiple operations. This IR can be used to chain together multiple operations in a sequence and/or to add loops around them. This is achieved through new and existing mlir operations.

#### `sdscbundle.sdsc_execute` op

The central operation in this IR is a new operation we introduce to instantiate a SuperDSC in the execution plan of the complex kernel. This is an example:

```mlir
sdscbundle.sdsc_execute (%A_start_address, %B_start_address) {sdsc_filename="sdscA.json", symbol_ids=[-1, -2]}
```

The operation does not **return** anything.

The operation **attributes** are:
* `sdsc_filename` the relavite path and filename of the specific sdsc.json to instantiate, relative to the location of the mlir file. As one bundle can contain multiple sdsc, different names must be chosen for each json file and here we can refer to the exact one of interest.
* `symbol_ids` list of the symbol ids used inside the sdsc, if any, to represent symbolic start addresses or sizes

The operation **operands** are the SSA variables corresponding to the values to be assigned to the symbols listed in `symbol_ids`, passed in the same order as the symbol ids. These values can be constants (`arith.constant`), or the result of affine expressions, like [`affine.apply`](https://mlir.llvm.org/docs/Dialects/Affine/#affineapply-affineaffineapplyop). The affine expressions can be comprised of constants and loop iterators. Having actual symbols in mlir will be supported through the next revision of the spec.

The `symbols_ids` must be unique in the bundle i.e., symbols ids cannot be recycled across sdscs that are part of the same bundle (unless they take the same value).

#### `sdscbundle.device_mem_allocate` op

A bundle often needs device memory that is not one of the kernel's inputs or outputs: buffers holding intermediate tensors handed from one SDSC to the next, and scratch space used within an SDSC. `sdscbundle.device_mem_allocate` lets the frontend request such memory from the backend, instead of keeping its address symbolic and having it supplied to the bundle from outside. The requested memory is held for the full `SuperDSC-Bundle`: it is reserved before the first SDSC in the bundle executes and is not deallocated at any intermediate point within the bundle.

```mlir
%pool = sdscbundle.device_mem_allocate 65536 bytes : index
```

The operation **returns** the byte address of the first byte of the allocated buffer, as an `index`. This is a device address in the same address space as the start addresses used inside `sdsc.json`, so it can be passed to `sdscbundle.sdsc_execute` as the value of a symbolic start address.

The operation **attributes** are:
* `size` the size of the requested buffer in bytes. It must be a positive constant; symbolic sizes are not supported. A request can be for up to ~15GB: the memory is carved out of a single memory segment whose maximum size is 16GB, of which 1GB is reserved for backend-generated programs and correction-related tensors.

The operation takes no **operands**.

The requested memory is a single contiguous range, and its contents are undefined at allocation.

##### Lifetime

The allocation lives for the entirety of the kernel. The backend reserves the requested bytes for the whole job, so the buffer is valid from the start of the bundle through the end of its last SDSC, and is released only once the kernel completes. There is no matching deallocate operation, and the backend does not reuse the range for anything else within the kernel. Consequently:
* the operation should appear in the entry block of the bundle function, outside of any `scf.for`. An allocation written inside a loop still reserves one single buffer, not one buffer per iteration.
* each `device_mem_allocate` in the bundle gets its own non-overlapping range, so the device memory a bundle requires is the sum of all of its requests, and it is that sum which must stay within the ~15GB budget above. A frontend that wants to reuse space across tensors with non-overlapping live ranges should request a single pool and sub-allocate it itself, as described below.

##### Sub-allocation

Individual buffer addresses are derived from the returned base address by adding a constant offset (`arith.addi`), and the result is passed to `sdscbundle.sdsc_execute` in place of an absolute start address. When a pool is carved up this way the frontend owns its layout: it must ensure that buffers whose live ranges overlap are given disjoint offset ranges, and that each offset satisfies the alignment required by the tensor placed there.

This example allocates a 64KB pool and splits it into four 16KB buffers, reusing the first two as inputs of a later SDSC:

```mlir
%pool = sdscbundle.device_mem_allocate 65536 bytes : index

%off_0     = arith.constant 0 : index
%off_16384 = arith.constant 16384 : index
%off_32768 = arith.constant 32768 : index
%off_49152 = arith.constant 49152 : index

%addr_0     = arith.addi %pool, %off_0     : index   // sdsc_0 output, sdsc_2 input
%addr_16384 = arith.addi %pool, %off_16384 : index   // sdsc_1 output, sdsc_2 input
%addr_32768 = arith.addi %pool, %off_32768 : index   // sdsc_2 output, sdsc_3 input
%addr_49152 = arith.addi %pool, %off_49152 : index   // sdsc_3 scratch

sdscbundle.sdsc_execute (%arg_0, %addr_0) {sdsc_filename="sdsc_0.json", symbol_ids=[-1, -2]}
sdscbundle.sdsc_execute (%arg_1, %addr_16384) {sdsc_filename="sdsc_1.json", symbol_ids=[-3, -4]}
sdscbundle.sdsc_execute (%addr_0, %addr_16384, %addr_32768) {sdsc_filename="sdsc_2.json", symbol_ids=[-5, -6, -7]}
sdscbundle.sdsc_execute (%addr_32768, %addr_49152, %arg_2) {sdsc_filename="sdsc_3.json", symbol_ids=[-8, -9, -10]}
```

#### Loops
Loops are represented using [`scf.for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop) operation borrowed from MLIR's SCF dialect. This allows SuperDSC-Bundle to describe complex kernels with multiple levels of loops and multiple SDSCs.

We do not support loop carried variables, the only supported scenario is the direct use of the induction variable (loop iterator).

The loop bound should be a constant. Symbolic loop bound will be enhanced in the next revision of the spec.

### `sdsc.json` filling

Each `sdsc_*.json` file describes one Spyre operation. The nesting is not obvious
from a field-by-field reference, so the overall shape is given here; the
field-level contract, including every constraint on how these fields are filled,
is in [`spec/json/`](spec/json/).

```
{
  "<operation name>": {                     // SuperDsc
    "numCoresUsed_":  <int>,
    "coreFoldProp_":     { "factor_": <int>, "label_": "core" },
    "coreletFoldProp_":  { "factor_": 2,     "label_": "corelet" },
    "dscs_": [                              // one entry per work division
      { "<dsc name>": {                     // DesignSpaceConfig
          "computeOp_":  [ { "opFuncName": "<op>", "attributes_": { ... } } ],
          "labeledDs_":  [ { ... } ],       // tensors
          "primaryDsInfo_":   { "<ROLE>": { "stickDimOrder_": [...], ... } },
          "N_":               { ... },      // total dims, all cores
          "dataStageParam_":  { "0": { "name_": "core", "ss_": {...}, "el_": {...} } },
          "constantInfo_":    { "0": { "name_": "...", ... } },
          "scheduleTree_":    [ { "nodeType_": "allocate", ... } ]
      } }
    ]
  }
}
```

A `dscs_` entry is an array element whose value is an object keyed by an arbitrary
DSC name — both levels are required. With balanced work division a single entry is
enough. `DesignSpaceConfig` represents both deep-learning operators and
data-shuffle operations (stick-breaking, non-stick-breaking, gather, scatter).

Where to find each part of the contract:

| To fill | See |
|---|---|
| The operation and its attributes | [ComputeOperation](spec/json/computeoperation.md) |
| Core counts and fold factors | [FoldProperty](spec/json/foldproperty.md), [FoldManager](spec/json/foldmanager.md) |
| Work division across cores | [DesignSpaceConfig](spec/json/designspaceconfig.md) |
| Per-core tile sizes | [DataStageParam](spec/json/datastageparam.md), [DataStructDims](spec/json/datastructdims.md) |
| Tensors, formats and `scale_` | [LabeledDataStructure](spec/json/labeleddatastructure.md) |
| Stick and memory layout | [PrimaryDsInfo](spec/json/primarydsinfo.md), [Stick Layout Constraints](spec/json/stick-layout-constraints.md) |
| Padding for window operations | [Padding](spec/json/padding.md) |
| Allocations, addresses and gaps | [ScheduleTreeNode](spec/json/scheduletreenode.md) |
| Coordinates | [CoordinateContainer](spec/json/coordinatecontainer.md), [CoordinateInfo](spec/json/coordinateinfo.md) |
| Constants | [ConstantInfo](spec/json/constantinfo.md) |
| Symbolic dimensions | [DesignSpaceConfig](spec/json/designspaceconfig.md), [DataStructDims](spec/json/datastructdims.md) |

#### Coordinate arrangement

Tensor coordinates per dimension are expressed as a sequence of nested affine
expressions, `alpha * index + beta`, in `allocateCoordinates_`. The combined
factor across all folds must equal the number of elements in that dimension.

The following illustrates the encoding and is not itself normative. For the
coordinate sequence

    0, 1, 2, 3, 64, 65, 66, 67, 4, 5, 6, 7, 68, 69, 70, 71

the arrangement, outermost to innermost, is

| alpha | beta | factor |
|---|---|---|
| 4 | 0 | 2 |
| 64 | 0 | 2 |
| 1 | 0 | 4 |

Tensor allocation need not match the compute work division: data resident on one
core may be consumed by an operation on another, and the backend inserts the
required movement.

### Operations, precisions and constants

The permitted values of `opFuncName`, the precision each operation runs in, and the
constants each requires are specified in
[Supported Operations](spec/json/computeoperation.md#supported-operations).

### Layout and work-division constraints

Stick-layout constraints per operation category, and the constraints on how work may
be divided across cores, are specified in
[Stick Layout Constraints](spec/json/stick-layout-constraints.md).

## Examples

Worked examples, in increasing order of complexity, are in [`examples/`](examples/).
They illustrate the format and are not themselves normative.
