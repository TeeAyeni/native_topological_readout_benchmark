# QWC Hamiltonian Public Code Subset

This folder contains the self-contained public release for the QWC Hamiltonian study. The layout is organized around what a new user actually needs to do:

- run a workflow
- reduce or analyze outputs
- generate figures

Included capabilities:

- noiseless Floquet-evolution FR-vs-PS_QWC runs
- IBM hardware submission for the Floquet-evolution workflow
- noiseless locked-state VQE FR-vs-PS_QWC runs
- IBM hardware submission for the locked-state VQE workflow
- reduction of returned IBM job manifests into JSONL datasets
- analysis of JSONL datasets into summary JSON
- manuscript-style figure generation from four summary JSON inputs
- bundled example summary JSONs for a self-contained plotting demo


## Layout

```text
github_public/
  workflows/
    workflow_support.py
    floquet_evolution/
      run_noiseless_floquet_evolution.py
      submit_floquet_hardware_jobs.py
    locked_state_vqe/
      run_noiseless_locked_state_vqe.py
      submit_locked_state_vqe_hardware_jobs.py
  analysis/
    collect_ibm_job_results.py
    summarize_benchmark_dataset.py
    generate_manuscript_figures.py
  dependencies/VQE_FR_PS/
    anyon_vqe.py
    locked_state_payloads/
  examples/summaries/
  run_workflow.py
  publish_preflight.py
  requirements.txt
```

## Environment

Install the pinned minimal stack:

```bash
pip install -r requirements.txt
```

The versions were chosen from the validated archive environment.

## Repository Metadata

Publish-facing metadata drafts are included in this folder:

- `GITHUB_METADATA.md` for the repository description, topics, and release copy
- `RELEASE_NOTES_v1.0.0.md` for the initial public release notes
- `CITATION.cff` for software citation metadata

## Publish Preflight

Run the bundled preflight before uploading the folder or pushing it as a standalone repository:

```bash
python publish_preflight.py
```

This performs a minimal end-to-end offline check of the public subset:

- runs small Floquet and locked-state VQE noiseless jobs
- dry-runs both IBM hardware submission entrypoints
- analyzes the generated noiseless JSONL outputs
- generates figures from the bundled example summaries

Temporary outputs are removed automatically. Use `--keep-temp` if you want to inspect them.

## Convenience Wrapper

You can call the scripts directly, or use the top-level wrapper:

```bash
python run_workflow.py <command> [script arguments]
```

Available wrapper commands:

- `floquet-noiseless`
- `floquet-hardware-submit`
- `locked-state-vqe-noiseless`
- `locked-state-vqe-hardware-submit`
- `collect-ibm-job-results`
- `summarize-dataset`
- `generate-manuscript-figures`

Legacy command aliases are still accepted locally for compatibility, but the names above are the public interface.

## Example Commands

### 1. Floquet noiseless run

```bash
python run_workflow.py floquet-noiseless \
  --num-qubits 7 \
  --steps 1-6 \
  --budgets 2000,4000,8000,16000 \
  --replicates 5 \
  --output outputs/floquet_noiseless_hamiltonian.jsonl
```

### 2. Locked-state VQE noiseless run

```bash
python run_workflow.py locked-state-vqe-noiseless \
  --sizes 5,7,9,11 \
  --budgets 2000,4000,8000,16000 \
  --replicates 5 \
  --output outputs/locked_state_vqe_noiseless_hamiltonian.jsonl
```

### 3. Floquet hardware submission

```bash
python run_workflow.py floquet-hardware-submit \
  --backend ibm_pittsburgh \
  --num-qubits 7 \
  --steps 1,2,3,4,5,6 \
  --budgets 2000,4000,8000,16000 \
  --output outputs/floquet_hardware_manifest.json
```

Required environment variables for IBM Runtime access:

- `IBM_NAME`
- `IBM_CHANNEL`
- `IBM_INSTANCE`

Use `--dry-run` first to inspect the planned submission without sending jobs.

### 4. Locked-state VQE hardware submission

```bash
python run_workflow.py locked-state-vqe-hardware-submit \
  --backend ibm_pittsburgh \
  --sizes 5,7,9,11 \
  --budgets 2000,4000,8000,16000 \
  --output outputs/locked_state_vqe_hardware_manifest.json
```

### 5. Collect and reduce returned hardware jobs

The reducer expects the manifest written by one of the submit scripts.

```bash
python run_workflow.py collect-ibm-job-results \
  --manifest outputs/floquet_hardware_manifest.json \
  --mode floquet \
  --output outputs/floquet_hardware_hamiltonian.jsonl
```

```bash
python run_workflow.py collect-ibm-job-results \
  --manifest outputs/locked_state_vqe_hardware_manifest.json \
  --mode locked-state-vqe \
  --output outputs/locked_state_vqe_hardware_hamiltonian.jsonl
```

### 6. Summarize any JSONL dataset into a summary JSON

```bash
python run_workflow.py summarize-dataset \
  --dataset outputs/floquet_noiseless_hamiltonian.jsonl \
  --group-fields dataset_tag,num_qubits,num_steps,budget_total \
  --out-json outputs/floquet_noiseless_hamiltonian_summary.json
```

```bash
python run_workflow.py summarize-dataset \
  --dataset outputs/locked_state_vqe_noiseless_hamiltonian.jsonl \
  --group-fields dataset_tag,num_qubits,budget_total \
  --out-json outputs/locked_state_vqe_noiseless_hamiltonian_summary.json
```

```bash
python run_workflow.py summarize-dataset \
  --dataset outputs/floquet_hardware_hamiltonian.jsonl \
  --group-fields dataset_tag,num_qubits,num_steps,budget_total \
  --out-json outputs/floquet_hardware_hamiltonian_summary.json
```

```bash
python run_workflow.py summarize-dataset \
  --dataset outputs/locked_state_vqe_hardware_hamiltonian.jsonl \
  --group-fields dataset_tag,num_qubits,budget_total \
  --out-json outputs/locked_state_vqe_hardware_hamiltonian_summary.json
```

### 7. Generate the manuscript-style figure bundle

```bash
python run_workflow.py generate-manuscript-figures \
  --date-tag public \
  --floquet-noiseless-summary outputs/floquet_noiseless_hamiltonian_summary.json \
  --floquet-hardware-summary outputs/floquet_hardware_hamiltonian_summary.json \
  --locked-state-vqe-noiseless-summary outputs/locked_state_vqe_noiseless_hamiltonian_summary.json \
  --locked-state-vqe-hardware-summary outputs/locked_state_vqe_hardware_hamiltonian_summary.json \
  --output-dir outputs/manuscript_figures_public
```

### 8. Plot directly from the bundled example summaries

```bash
python run_workflow.py generate-manuscript-figures \
  --date-tag example \
  --floquet-noiseless-summary examples/summaries/floquet_noiseless_hamiltonian_summary.json \
  --floquet-hardware-summary examples/summaries/floquet_hardware_20260515_summary.json \
  --locked-state-vqe-noiseless-summary examples/summaries/locked_state_vqe_noiseless_hamiltonian_summary.json \
  --locked-state-vqe-hardware-summary examples/summaries/locked_state_vqe_hardware_20260515_summary.json \
  --output-dir outputs/manuscript_figures_example
```

## Notes

- The locked-state VQE workflows rely on the bundled `locked_state_payloads/` JSON files.
- The example summaries are included so the plotting workflow can be demonstrated without reaching outside this folder.
- The hardware reducer is generic to the manifest shape written by the bundled submission scripts.
- Hardware outputs will vary across calibration windows and backend revisions.
- For publication, upload the contents of this folder as the repository root; do not publish generated `outputs/` data or any IBM credential values.