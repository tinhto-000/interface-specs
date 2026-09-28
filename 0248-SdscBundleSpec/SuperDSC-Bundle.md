# SuperDSC-Bundle Interface Specification

**Status:** Active. The version of this contract is the backend compiler release it
ships with; see [JSON Schema](spec/json/SDSC-json-api.md).

**Authors:** @lupalby, @Prasanth-Chatarasi, @bmahjour, @viji560, @vswagath1989, @jneves60

## Summary

This document specifies the `SuperDSC-Bundle`, the interface between the torch-spyre
frontend compiler and the Spyre backend compiler (Deeptools).

## Scope

A SuperDSC-Bundle is the unit the frontend hands to the backend for one complex
kernel. This specification defines two things:

- the **SDSC JSON document format** — one document per Spyre operation, describing
  the operation, its tensors, and how its work is divided across cores;
- the **`sdscbundle` MLIR dialect** — the operations that sequence those documents,
  allocate device memory, and bind runtime values.

It defines what a producer may write and what a consumer accepts. It does not specify
how either compiler is implemented, how work should be divided for performance, or
how to author MLIR beyond the constraints stated here.

**Audience.** A reader implementing a producer of bundles, or a consumer of them.
Familiarity with MLIR and with the Spyre data-parallel execution model is assumed;
[`guide/`](guide/) is introductory and forms no part of this specification.

**Conformance** is defined in [SDSC JSON API](spec/json/SDSC-json-api.md): a
conforming document validates against
[`sdscbundle-schema.json`](spec/sdscbundle-schema.json) and satisfies the constraints
stated on the object pages that the schema cannot express.

## Normative references

| Reference | Role |
|---|---|
| [`spec/sdscbundle-schema.json`](spec/sdscbundle-schema.json) | The machine-readable contract for the JSON document format. |
| [JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12) | The dialect the schema is written in. |
| [MLIR SCF](https://mlir.llvm.org/docs/Dialects/SCFDialect/), [Arith](https://mlir.llvm.org/docs/Dialects/ArithDialect/), [Affine](https://mlir.llvm.org/docs/Dialects/Affine/) | Upstream operations permitted in a bundle, with the restrictions in [MLIR Bundle API](spec/dialect/MLIR-bundle-API.md). |
| [torch-spyre](https://torch-spyre.readthedocs.io/) | The frontend that produces bundles, and the torch operations it supports. |

## Terms

| Term | Meaning |
|---|---|
| **SuperDSC** | The top-level object of an SDSC JSON document, describing one Spyre operation and its mapping across cores. |
| **DSC** (DesignSpaceConfig) | One work division of a SuperDSC. A SuperDSC carries an array of them; with balanced work division a single entry suffices. |
| **core** | A RaPiD core: one compute engine plus its LX scratchpad. Cores are indexed from `0`. |
| **corelet** | A subdivision within a core. The corelet fold factor is `2`. |
| **fold** | One level of subdivision of a dimension, with a `factor_` giving the number of slices at that level. Folds nest, and their factors multiply to the dimension's element count. |
| **work division** | The assignment of an operation's dimension slices to cores, expressed by `numWkSlicesPerDim_` and `coreIdToWkSlice_` on the SuperDSC. |
| **stick** | The innermost contiguous group of elements transferred and operated on as a unit, given by `stickDimOrder_` and `stickSize_` in a `primaryDsInfo_` entry. |
| **datastage** | Tile dimensions at one level of the hierarchy — a `dataStageParam_` entry, holding steady-state (`ss_`) and epilogue (`el_`) dimensions. |
| **symbolic value** | A value not fixed at compile time — a tensor start address or a dimension size — resolved before the job is launched. |

## Execution model

The figure below illustrates what is called the Spyre Stack, where a user-written PyTorch program is compiled by torch-spyre to generate a set of files known as the **SuperDSC-Bundle**. A PyTorch file may translate into several SuperDSC-Bundles, each one composed of a `bundle.mlir` file and several `sdsc_*.json` files. Each SuperDSC-Bundle is compiled by Deeptools to generate the assembly code and execution plan that runs on the Spyre AI Accelerator Card.

<p align="center">
  <img src="guide/figures/torch_spyre_backend_flow.png" alt="torch_spyre_backend_flow" width="650"/>
</p>
<p align="center">
  Figure 1. High-level view of SuperDSC-Bundle API within the torch-spyre stack
</p>

`SuperDSC-Bundle` views the Spyre hardware at the data-parallel level of hardware abstraction. In this abstraction, Spyre is viewed as having multiple cores, with each core having a compute engine and a scratchpad memory. The cores are interfaced with each other and off-chip memory banks using an on-chip interconnect fabric.

<p align="center">
  <img src="guide/figures/data_parallel_hw_abstraction.png" alt="data_parallel_hw_abstraction" width="450"/>
</p>
<p align="center">
  Figure 2. Hardware abstraction of multi-core accelerator embodied in `SuperDSC-Bundle`
</p>

`SuperDSC-Bundle` enables the frontend compilers to express data-parallel mappings of:
* Complex kernels comprised of a sequence of operations
* The work division (or computation split) across RaPiD cores of Spyre for each operation
* The placement of input/output tensors to each operation either in DDR memory or LX scratchpad of the RaPiD cores
* Shapes of the operations and tensors is allowed to be static or symbolic
* The start address of the tensors in DDR and LX is allowed to be a fixed number or symbolic

The `SuperDSC-Bundle` specification is used by the Deeptools backend compiler to produce `SpyreCode` containing the job binary, a job plan and other compiled artifacts. In the scenario where either the start-address and/or shapes are symbolic, `SpyreCode` allows for the program binary to contain variables that need to be substituted or corrected before execution. The mechanism to effect program correction just-in-time before the job is launched onto Spyre is also produced by the backend compiler as part of `SpyreCode`.

### Related interfaces

* Frontend/Backend compiler interface will transition to a new interface called Kernel Tile Intermediate Representation (KTIR) in the future (https://github.com/torch-spyre/rfcs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)
* `SuperDSC` (without bundle capability) is the current interface between Deeptools frontend compiler and Deeptools backend compiler
* `SpyreCode` is tracked through: https://github.com/torch-spyre/torch-spyre/issues/277


## Specification contents

The normative contract is in [`spec/`](spec/) — the JSON document format under
[`spec/json/`](spec/json/) and the MLIR dialect under
[`spec/dialect/`](spec/dialect/). This document is the entry point; the pages below
are the specification.

**Start here**

| Page | Covers |
|---|---|
| [SDSC JSON API](spec/json/SDSC-json-api.md) | File structure, conformance, and the step-by-step filling procedure |
| [Object Hierarchy](spec/json/JSON-object-Hierarchy.md) | How the objects nest |
| [`sdscbundle-schema.json`](spec/sdscbundle-schema.json) | The machine-readable contract |

**The `sdscbundle` MLIR dialect**

| Page | Covers |
|---|---|
| [MLIR Bundle API](spec/dialect/MLIR-bundle-API.md) | Bundle container, `sdsc_execute`, `device_mem_allocate`, `input_arg_extract`, symbolic values, and the upstream MLIR operations permitted in a bundle |

**JSON objects, by role**

| Role | Pages |
|---|---|
| Top level | [SuperDsc](spec/json/superdsc-object.md), [DesignSpaceConfig](spec/json/designspaceconfig.md) |
| Tensors | [LabeledDataStructure](spec/json/labeleddatastructure.md), [MemoryOrganization](spec/json/memoryorganization.md), [PrimaryDsInfo](spec/json/primarydsinfo.md), [DataStructDims](spec/json/datastructdims.md) |
| Folding | [FoldProperty](spec/json/foldproperty.md), [FoldManager](spec/json/foldmanager.md) |
| Layout | [Padding](spec/json/padding.md), [Stick padding](spec/json/stick-padding.md), [Stick Layout Constraints](spec/json/stick-layout-constraints.md) |
| Execution | [DataStageParam](spec/json/datastageparam.md), [ScheduleTreeNode](spec/json/scheduletreenode.md), [CoordinateContainer](spec/json/coordinatecontainer.md), [CoordinateInfo](spec/json/coordinateinfo.md) |
| Operations | [ComputeOperation](spec/json/computeoperation.md), [ConstantInfo](spec/json/constantinfo.md) |
| | [Reference documents](spec/reference.md) |

Non-normative introductory material, worked examples and tuning guidance are in
[`guide/`](guide/) and form no part of the contract.

## Structure of SuperDSC-Bundle

The backend expects the frontend to produce multiple output files that work in conjunction to instruct the backend on how one or multiple programs can be compiled and executed in a sequence as part of a complex kernel. The expectation is to receive:
* one or more `sdsc_*.json` files, each describing a Spyre operation
* one `bundle.mlir` file with the SuperDSC-Bundle IR

### API Components

The API consists of two primary components:

1. **MLIR Bundle File** (`.mlir`) — Orchestrates execution flow, symbol management, and operation sequencing across one or more SDSC JSON files. See [MLIR Bundle API](spec/dialect/MLIR-bundle-API.md) for the full dialect reference.
2. **SDSC JSON Files** (`.json`) — Each file defines a single operation and its core mapping. See [SDSC JSON API](spec/json/SDSC-json-api.md) for the step-by-step filling guide.

All `sdsc_*.json` files must conform to the [SDSC Bundle JSON Schema](spec/sdscbundle-schema.json). The schema is the normative reference for structural correctness — it enforces required fields, enum values, and type constraints at every level of the object hierarchy. Semantic constraints (cross-field consistency) are described in the individual object pages linked from [SDSC JSON API](spec/json/SDSC-json-api.md).

### SuperDSC JSON Structure

SuperDSC is a self-contained compiled artifact that describes everything the Spyre hardware needs to execute a single scheduled operation deterministically. The top-level structure contains core fold properties, work-slice mappings, and a per-core execution schedule. A `dscs_` array holds one or more `DesignSpaceConfig` entries, each being a complete description of one compute configuration. See the [JSON Object Hierarchy](spec/json/JSON-object-Hierarchy.md) for the full object tree.

Each `DesignSpaceConfig` entry contains the following elements:

- **Core fold properties** (`coreFoldProp_`, `numWkSlicesPerDim_`, `coreIdToWkSlice_`): how to divide the iteration space across 32 cores. For a tensor of shape (1024, 256), this encodes how many rows each core processes. The encoding gives each core an equal number of sticks and keeps each core within its addressable device memory limit.
- **Tensor descriptors** (`labeledDs_`, `primaryDsInfo_`): for each tensor argument, the tiling structure defines which dimensions are stick dimensions, how the host-side shape maps to device-side tiles, memory residency (HBM vs. LX scratchpad), data format, and which dimensions each tensor iterates over fully vs. which are summed over (contracted) as in the K dimension of a matmul.
- **Schedule tree** (`scheduleTree_`): a list of allocate nodes (one per tensor) that specify memory placement (HBM or LX scratchpad), dimension ordering, per-core start addresses via fold mappings, and coordinate information encoding how each dimension is split across cores with affine transformations.
- **Data staging** (`dataStageParam_`): per-core dimension sizes for steady-state and epilogue passes, describing how data is partitioned for transfer into scratchpad.
- **Compute operations** (`computeOp_`): one entry per operation, encoding the execution unit (PT or SFP), operation name, data format, fidelity, and the input/output tensor references from `labeledDs_`.

**Folding** is a central concept in SuperDSC. A single parameterized artifact can represent multiple execution variants across time steps and cores without recompilation. Fold properties use affine transformations (`alpha * index + beta`) to compute per-core coordinates and addresses, so one JSON file describes the behavior of all 32 cores compactly instead of duplicating the description for each core. See [FoldProperty](spec/json/foldproperty.md) and [FoldManager](spec/json/foldmanager.md) for details.

### Important Notes

**DesignSpaceConfig can represent BOTH deep learning operators AND data-shuffle operations:**

- **Deep learning operators**: Matmul, convolution, activations, reductions, etc.
- **Data-shuffle operations**: Stick-breaking, non-stick breaking, gather, scatter

**Tensor allocation need NOT be compatible with compute work division.** Data in one core can be directly available for compute in another core — the backend compiler will ensure proper data movement across cores. This functionality is fully supported by the backend.


## Where each part of the contract lives

| To fill | See |
|---|---|
| The operation and its attributes | [ComputeOperation](spec/json/computeoperation.md) |
| Core counts and fold factors | [FoldProperty](spec/json/foldproperty.md), [FoldManager](spec/json/foldmanager.md) |
| Work division across cores | [SuperDsc](spec/json/superdsc-object.md), [DesignSpaceConfig](spec/json/designspaceconfig.md) |
| Per-core tile sizes | [DataStageParam](spec/json/datastageparam.md), [DataStructDims](spec/json/datastructdims.md) |
| Tensors, formats and `scale_` | [LabeledDataStructure](spec/json/labeleddatastructure.md) |
| Stick and memory layout | [PrimaryDsInfo](spec/json/primarydsinfo.md), [Stick Layout Constraints](spec/json/stick-layout-constraints.md) |
| Padding for window operations | [Padding](spec/json/padding.md), [Stick padding](spec/json/stick-padding.md) |
| Allocations, addresses and gaps | [ScheduleTreeNode](spec/json/scheduletreenode.md) |
| Coordinates | [CoordinateContainer](spec/json/coordinatecontainer.md), [CoordinateInfo](spec/json/coordinateinfo.md) |
| Constants | [ConstantInfo](spec/json/constantinfo.md) |
| `opFuncName` values, precisions and required constants | [ComputeOperation](spec/json/computeoperation.md) |
| Per-operation stick constraints and core work division | [Stick Layout Constraints](spec/json/stick-layout-constraints.md) |

The step-by-step procedure for filling a document is in
[SDSC JSON API](spec/json/SDSC-json-api.md).

## Examples

Worked examples are in [`guide/`](guide/) and [`examples/`](examples/). They
illustrate the format and are not normative.
