# Artifical Intelligence Protein Profiling: Command-line Inference Interface

Command-line interface for running AiPP residue-level predictions
(SSBind, LigBind, ZNBind, LigCys) on protein sequences using ESM-C
embeddings and pre-trained weights.

This repository is designed for use in publications and automated
pipelines (e.g. Zenodo-backed archives).

---

## Repository layout

- aippCLI.py
  Main command-line interface.

- env/wizard.sh
  Simple installation wizard that creates a Python virtual environment,
  installs dependencies, and downloads pretrained weights from Zenodo.

- env/wts/
  Default location for model weight directories. The CLI expects
  task-specific subdirectories here (e.g. ssbind_v1, ligbind_v1, etc.).

---

## Requirements

- NVIDIA GPU with >= 24GB VRAM (e.g. RTX 4090)
- 128 GB system memory
- Python 3.10 or newer
- POSIX-like environment (Linux / macOS)
- Packages:
  - numpy
  - torch
  - esm
  - tqdm
  - httpx
  - colorama

You can install these manually:

    pip install numpy torch esm tqdm colorama httpx

or use the provided wizard.

---

## Installation

Clone the repository:

    git clone https://github.com/wayyne/aippCLI.git
    cd aippCLI

Run the installation wizard:

    bash env/wizard.sh

The wizard:

- creates a virtual environment named "AiPP"
- activates it
- upgrades pip
- installs core Python dependencies
- is the place to add commands to download weights from Zenodo into
  env/wts/

To re-activate the environment later:

    source AiPP/bin/activate

---

## Weights

By default, aippCLI.py looks for model weights under:

    env/wts/

with task-specific subdirectories (for example):

- env/wts/ssbind_v1
- env/wts/ligbind_v1
- env/wts/znbind_v1
- env/wts/ligcysA_v1
- env/wts/ligcysS_v1

You can override the root directory for weights with:

    export AIPP_WTS_DIR=/path/to/wts

and the CLI will use that directory instead of env/wts/.

---

## Forge token

The CLI requires an ESM Forge token to compute ESM-C embeddings.

Token handling:

1. First run:
   - Provide a token via --forge-token, either as the raw string or a
     path to a file that contains the token.
   - The token is cached to a user-level file (by default:
     ~/.aipp_forge_token).

2. Subsequent runs:
   - If --forge-token is omitted, the cached token is used.
   - If no cached token exists, the CLI will prompt for one
     interactively (input is hidden) and then cache it.

You can override the cache location with:

    export AIPP_FORGE_TOKEN_FILE=/path/to/token_file

---

## Basic usage

Activate the environment:

    source AiPP/bin/activate

Run a single sequence:

    python aippCLI.py \
      --sequence "ACDEFGHIKLMNPQRSTVWY" \
      --id example1 \
      --forge-token /path/to/forge_token.txt

Run multiple sequences from a FASTA file:

    python aippCLI.py \
      --fasta proteins.fasta \
      --forge-token /path/to/forge_token.txt

If a token has already been cached, you can omit --forge-token and the
CLI will reuse the saved token or prompt for one if needed.

---

## Output

By default, per-residue predictions are printed as a tab-separated
table to standard output.

To write the table to a file:

    python aippCLI.py \
      --sequence "ACDEFGHIKLMNPQRSTVWY" \
      --id example1 \
      --out results.tsv

At the end of the run you will see:

    output written to: results.tsv

The first line of the file is a header:

    pos    AA    SSBind    SSBind_topN    LigBind    LigBind_topN ...
    ...

Each subsequent line corresponds to a residue position.

---

## Reproducibility

The CLI echoes the full command line used to invoke it immediately
after the splash screen. This makes it straightforward to record and
reproduce runs from logs or publications.

For publication or archival use (e.g. Zenodo), store:

- the exact repository commit
- the weights archive (Zenodo DOI)
- the command line used for each run

