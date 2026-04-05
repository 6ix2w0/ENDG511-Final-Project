# Fish AI (Underwater) — FishNet taxonomy

This repo builds an AI tool that classifies fish images into **family / genus / species** (top-100 species initially), with **Unknown / Not sure** via confidence thresholding. It compares several learning setups in **Jupyter notebooks**: SSL pretrain + finetune, few-shot, federated (simulated FedAvg), and optional model compression.

## Repo layout

- `notebooks/`: data prep, each learning method, comparisons, demo UI
- `src/fish_ai/`: datasets, models, training, evaluation, compression
- `configs/`: experiment configs (YAML)
- `data/`:
  - `raw/`: downloaded datasets (not committed)
  - `interim/`, `processed/`: optional intermediates (not committed)
  - `manifests/`: JSONL/CSV manifests used by training code

## Notebooks

- `00_setup.ipynb` — install deps, device check
- `01_data_build_manifests.ipynb` — FishNet → taxonomy manifests
- `02_ssl_pretrain_fish_only.ipynb` — SimCLR on fish images
- `03_ssl_finetune_taxonomy.ipynb` — fine-tune taxonomy heads
- `04_fewshot_taxonomy.ipynb` — k-shot runs
- `05_federated_taxonomy_fedavg.ipynb` — simulated FedAvg
- `06_compare_all_methods.ipynb` — plots / tables (regenerate via `python scripts/gen_nb06.py`)
- `07_gradio_demo.ipynb` — local taxonomy demo (full image)
- `08_model_compression.ipynb` — prune → INT8 → Huffman-style blob (taxonomy checkpoints)

## Sources

When external code or dataset tooling is reused/adapted, record the **source link** in `SOURCES.md` and at the top of the file that contains the reused code.
