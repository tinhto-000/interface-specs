# Conformance

A conforming SDSC document validates against
[`sdscbundle-schema.json`](../sdscbundle-schema.json), written against
[JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12). Validating
against that file is the definition of conformance for the JSON layer of a
SuperDSC-Bundle.

Each `sdsc_*.json` file in a bundle is validated independently. The MLIR file that
references them is covered by [MLIR Bundle API](../dialect/MLIR-bundle-API.md).

## What validation rejects

| Category | Rejected when |
|---|---|
| Required fields | a field listed as required for an object is absent |
| Types | a value has the wrong JSON type. Fields typed `integer` reject a fractional number such as `1.5`; fields typed `number` accept it |
| Enum values | a string falls outside the set listed for that field |
| Numeric bounds | a value falls outside the declared minimum or maximum |
| Unknown fields | an object contains a field name not defined for it (see below) |
| Dynamic keys | a map key does not match the pattern declared for it |

### Objects are closed

Every object in the schema except `DataStructDims` and `DimPaddingSizes` declares
`additionalProperties: false`. A field name that is not defined for such an object
is an error, not an ignored extra.

The practical consequence: a misspelled key does not silently do nothing — it
fails validation.

Trailing underscores are the most common source of this. They are **not** uniform
across the format: most field names end in `_`, but around thirty do not, and there
is no rule that predicts which. Do not infer a field name — take it from the field
page for the object you are emitting, or from the schema.

### Dynamic keys

Several objects are maps whose keys a producer chooses. The permitted key shapes
are:

| Key shape | Used for |
|---|---|
| `^[A-Za-z_][A-Za-z0-9_]*$` | dimension names — `numWkSlicesPerDim_`, `paddingSizes_`, `backGapCore_`, `dimToSymbolMapping_`, `symbolicDimInfo_` |
| `^[0-9]+$` | non-negative integer ids as decimal strings — `coreIdToWkSlice_`, `dataStageParam_`, `constantInfo_`, `coreIdToDsc_` |
| `^-?[0-9]+$` | core ids that may be negative — the inner key of `backGapCore_` |
| `^[A-Z_]+$` | tensor role labels — `primaryDsInfo_` |

A numeric map key is a **quoted string** in JSON: `"0"`, not `0`.

## Constraints the schema file cannot express

Not every rule in this specification is machine-checkable. JSON Schema constrains
one value at a time: it can require that `factor_` is an integer of at least 1, but
it cannot multiply several `factor_` values together and compare the result to a
tensor's element count. Rules that relate fields to each other, or to a computed
size, are therefore stated in prose on the field pages rather than encoded in the
schema file.

A document that satisfies the schema file but violates one of these rules is not a
conforming document. Both sets of rules are part of the contract.

Rules of this kind include:

- Combined fold factors, multiplied together, must equal the number of elements in
  that dimension for the tensor.
- `scale_` and `maxDimSizes_` are positional — their order must match
  `layoutDimOrder_`.
- `stickSize_` is parallel to `stickDimOrder_`, and `stickRepl_` to both.
- When a symbolic dimension is divided across cores, the number of slices must
  divide the granularity.
- Padding sizes must be consistent between the total-size dimensions and the
  per-core datastage.

Each is stated on the field page for the field it constrains. The list above is
illustrative, not exhaustive — the field pages are authoritative.

## Compatibility

This specification and the compiler that implements it are released together, so the
backend compiler's **release tag** is the version of this contract. The schema
records it in its `deeptoolsRelease` field — see
[JSON Schema](json-schema.md#which-release-this-schema-describes) — so a copy of the
schema on its own still says which release it describes.

Cite that tag when naming a version of this specification, and validate against the
schema from the release you are targeting.

| Change between releases | Effect on an existing document |
|---|---|
| A field, object, or permitted value is added | remains conforming |
| A field's type, bound, or required-ness is narrowed, or a permitted value is withdrawn | may stop conforming; re-validate against the newer schema |

---

[↑ SuperDSC-Bundle specification](../../SuperDSC-Bundle.md)
