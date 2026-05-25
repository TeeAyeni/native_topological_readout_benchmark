# Release Notes

## v1.0.0

Release date: 2026-05-24

This is the initial public software release for the QWC Hamiltonian benchmark accompanying the PRA manuscript on native topological readout and measurement-compilation trade-offs for Fibonacci-chain Hamiltonians.

### Included in this release

- self-contained Floquet-evolution noiseless workflows
- self-contained locked-state VQE noiseless workflows
- IBM hardware submission scripts for both Floquet-evolution and locked-state VQE workloads
- manifest-based IBM job collection and reduction into JSONL datasets
- dataset summarization into summary JSON files
- manuscript-figure generation from the four summary inputs
- bundled example summaries for a self-contained plotting demo
- bundled locked-state payloads and VQE dependency code
- one-command publish preflight validation via `publish_preflight.py`

### Validation performed

- Floquet-evolution noiseless smoke run completed successfully
- locked-state VQE noiseless smoke run completed successfully
- Floquet-evolution hardware submission dry-run completed successfully
- locked-state VQE hardware submission dry-run completed successfully
- dataset summarization on generated smoke datasets completed successfully
- manuscript-figure generation from bundled example summaries completed successfully
- privacy and path sweep found no local absolute paths, username strings, legacy project-tree names, or email-like strings inside the public folder

### Notes and limits

- IBM Runtime credentials are not included; hardware submission requires `IBM_NAME`, `IBM_CHANNEL`, and `IBM_INSTANCE` in the environment.
- Hardware outcomes are expected to vary across backend calibrations and execution windows.

### Release packaging note

- Generate any release-asset checksum from the final GitHub export or release zip at publish time so it matches the uploaded artifact exactly.