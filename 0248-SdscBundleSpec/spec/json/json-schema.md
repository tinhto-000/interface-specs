# JSON Schema

[`sdscbundle-schema.json`](../sdscbundle-schema.json) is the machine-readable contract for every
`sdsc_*.json` file in a SuperDSC-Bundle. It is written against
[JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12).

This file is the **input contract**: the fields a producer may set. It is derived
from the backend compiler's own schema but is deliberately narrower — the backend
must also accept documents it has itself written out, which carry fields no producer
ever sets. Those are excluded here.

Any divergence is intentional and justified by what a producer emits: a field the
frontend emits belongs in this file, whatever the backend does with it, and a field
only the backend writes does not.

## Which release this schema describes

The `deeptoolsRelease` field at the top of the schema names the backend compiler
release it was taken from — `v1.3.1`, for example. That tag is the version of this
contract.

The schema does not carry a version of its own. It is published alongside the
compiler that implements it and is never released on a separate cadence, so a second
numbering line would only drift against the release tags — which is what happened to
the private `version` field this replaced. To name a version of this specification,
cite the release tag; to check a document, validate against the schema from that
release.

When the schema is updated, update `deeptoolsRelease` to the release it came from.

The schema enforces:

- Required fields and their types at every level of the object hierarchy
- Enum values for `dsType_`, `component_`, `indirectAllocType_`, `indexTensorType_`,
  `dataFormat_`, `fidelity_`, `nodeType_`, and all other constrained string fields
- Pattern constraints on dynamic keys such as core IDs and operation-name root keys
- Structural rules such as `additionalProperties: false` on all defined objects

Semantic constraints that go beyond what JSON Schema can express — cross-field consistency,
stick-size alignment, and symbol-ID correspondence — are documented in the individual object
pages and in the [Validation section of SDSC JSON API](SDSC-json-api.md).


---

[↑ SuperDSC-Bundle specification](../../SuperDSC-Bundle.md)
