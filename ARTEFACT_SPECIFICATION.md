# Strict positive-witness candidate evidence artefact

The normative paired artefact schema is `schemas/pcpi_paired_candidate_certificate.schema.json`.
The legacy schema token `candidateStrictlyParetoNondominatedInMean` is retained for replay compatibility; normatively it denotes strict positive-witness archive separation. A conforming verifier validates the schema, semantic hash, raw-data hash,
sufficient statistics, covariance matrices, boundary parameters, witness
indices, stopping count and verdict.

The artefact establishes internal consistency only. External simulator
provenance and distributional adequacy remain protocol-level responsibilities.
