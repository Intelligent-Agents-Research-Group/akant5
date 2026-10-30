# AkanT5

Code for adapting the multilingual encoder–decoder mT5 (small, base, large) to Akan. The vocabulary
is either **replaced** with an Akan–English vocabulary of 64,100 entries or **extended** with Akan
subword units. The models are then trained with continued pretraining (CPT) on Twi text with 20%
English replay. The repository also contains the evaluation used in the accompanying paper:

- **Character-span bits per character (BPC):** reconstruction of masked character spans, normalized
  by the number of masked characters, so that models with different tokenizers are scored on the
  same text.
- **A frozen MT protocol:** supervised English–Twi fine-tuning with learning-rate selection on
  development data, three seeds, and paired bootstrap comparisons.

The adapted checkpoints and their vocabulary initializations are available on Hugging Face under
[IARG-UF](https://huggingface.co/IARG-UF), grouped in the
[AkanT5 collection](https://huggingface.co/collections/IARG-UF/akant5-cpt-checkpoints-and-vocabulary-initializations).
The collection contains six CPT checkpoints and six vocabulary initializations:

```text
IARG-UF/akant5-{small,base,large}-{replacement,extension}
IARG-UF/akant5-{small,base,large}-{replacement,extension}-init
```

The suffix `-init` identifies weights before CPT. Each repository includes the matching tokenizer.
**The model weights are released for research use only**,
because most of the CPT training corpora are licensed for non-commercial use (see
[Data](#data-and-licences)).

## Installation

Python 3.11 or later is required. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # installs PyTorch built for CUDA 12.8 on Linux
```

or with pip, after installing a PyTorch build for your platform:

```bash
pip install -e .
```

The command-line interface is `akanlm` (equivalently `python -m akanlm.cli`); `akanlm --help` lists
all commands. Commands read data and models from the Hugging Face cache and download what is
missing. To forbid downloads, set `HF_HUB_OFFLINE=1`; in such an environment,
`AKANLM_ALLOW_NETWORK=1` re-enables them for one command. FLORES+ is gated on the Hub: accept its
terms and set `HF_TOKEN`.

## Repository layout

```
src/akanlm/
  audit/      corpus declarations, manifest, n-gram decontamination
  corpora/    loaders, normalization, pinned column extraction
  data/       document splits, mixtures, packing, span-corruption collator
  tokenizer/  SentencePiece training, vocabulary extension, mT5 surgery, row initialization
  training/   continued pretraining (single process or DDP), validation, checkpoints
  eval/       character-span BPC, test-split BPC, MT probe and frozen MT protocol
  cli.py      command-line interface
configs/
  corpora/    one declaration per source, pinned to a Hub revision
  mixtures/   m1x (Twi) and m1x_en (Twi with 20% English replay)
  tokenizer/  32k, 48k and 64k replacement candidates; 64k candidates for extension
  models/     mT5-small/base/large, replacement and extension
  accel/      per-GPU batch layouts for 1, 2 and 4 GPUs
```

## Pipeline

Commands run from the repository root. Outputs go to `data/` and `runs/`.

The tokenizer-training commands below describe **fresh adaptation runs**. They fit both replacement
and extension candidate vocabularies on the training partition of a frozen CPT split. The paper's
released replacement tokenizer was fitted before the CPT partitions were constructed, so its
fitting input may include documents later assigned to CPT validation and test. The extension
candidate tokenizer was fitted on the frozen training partition. To preserve the vocabulary and
initialization used in the paper, reuse the corresponding released `-init` repository and its
tokenizer. Retraining with the fresh-run recipe can produce a different vocabulary even when its
size is unchanged.

### 1. Corpora

```bash
akanlm audit sources                                    # validate every declaration
```

Sources whose Hub files carry audio next to the transcripts (Kumawood, Ghana Speech, and others) and
the mC4 English replay are extracted once, with network access; only the declared text columns are
read:

```python
from akanlm.audit.sources import load_sources
from akanlm.corpora.extract import extract
from akanlm.data.mixture import derive_source_caps, load_all

sources = load_sources("configs/corpora")
caps = derive_source_caps(load_all("configs/mixtures"))
for name, source in sorted(sources.items()):
    if source.hf_files is not None and name in caps:
        extract(source, max_rows=caps[name].rows, max_chars=caps[name].chars)
# English used for tokenizer training, read up to the budget stated in its declaration:
extract(sources["english_replay_mc4"], max_rows=None, max_chars=83_109_901)
```

Then measure the corpora and build the decontamination ledger against the evaluation sets:

```bash
AKANLM_ALLOW_NETWORK=1 akanlm audit manifest
akanlm audit decontaminate
```

### 2. Document splits and tokenizers

```bash
akanlm cpt-split --mixture m1x_en --output data/manifests/m1x_en_split.json
# Replacement vocabulary (64,000 pieces + 100 sentinels)
akanlm tokenizer train -c configs/tokenizer/spm_64k.yaml --split-manifest data/manifests/m1x_en_split.json
# Candidate pieces for extension, then the extended tokenizer of each size
akanlm tokenizer train -c configs/tokenizer/spm_64k_train.yaml --split-manifest data/manifests/m1x_en_split.json
akanlm tokenizer extend --candidates runs/tokenizer/akant5_spm_64k_train_v1 \
    --base google/mt5-small --output runs/tokenizer/mt5_extended_small_64k_train_v1
```

For these fresh runs, tokenizers are fitted only on the training partition of the frozen split.

### 3. Initialization

```bash
# Replacement: new embedding and output matrices, rows initialized from mT5
akanlm model surgery --base google/mt5-small --tokenizer runs/tokenizer/akant5_spm_64k_chars_v3
# Extension: original rows kept, added rows initialized from mT5
akanlm model extend --base google/mt5-small \
    --tokenizer runs/tokenizer/mt5_extended_small_64k_train_v1 \
    --output runs/init/akant5-small-extended-64k-train-v1
```

### 4. Continued pretraining

Preparation packs the training data on CPU; training then reads the same cache. The accelerator
profile fixes the per-GPU batch, gradient accumulation and activation checkpointing, so that every
run uses a global batch of 1,024 sequences.

```bash
ARGS="--config configs/models/mt5-small.yaml --mixture m1x_en --single-arm \
      --split-manifest data/manifests/m1x_en_split.json --accel b200_2gpu \
      --output runs/cpt/akant5-small-m1x_en"
akanlm cpt $ARGS --prepare-only --workers 4                     # CPU
torchrun --standalone --nproc_per_node 2 -m akanlm.cli cpt $ARGS   # 2 GPUs
```

For extension, use `configs/models/mt5-<size>-extended.yaml` and the matching `*_extension`
accelerator profile. The run scores its validation split before and after training
(`evaluation/initialization_char_bpc.json`, `evaluation/final_char_bpc.json`).

### 5. Test-split BPC

```bash
akanlm eval test-bpc runs/cpt/akant5-small-m1x_en --checkpoint initialization
akanlm eval test-bpc runs/cpt/akant5-small-m1x_en --checkpoint final
```

The test split is scored with the weights, split, masking settings and scorer recorded for
validation.

### 6. Machine translation

The frozen protocol fine-tunes every model on MAFAND-MT English–Twi, selects the learning rate on
development chrF++ from a five-point grid, fine-tunes with three seeds at that rate, and reports
seed means with paired sentence-bootstrap intervals on FLORES+ (Akuapem, Asante) and MAFAND-MT.

```bash
STUDY=runs/eval/mt/main
# Freeze the protocol and register the systems. By default this also registers each CPT model's
# initialization, the mT5 checkpoint of the same size, and Afri-mT5-base.
akanlm eval mt-freeze --study $STUDY --dropout-rate 0.1 runs/cpt/akant5-small-m1x_en/final
# For every registered system (model path or Hub id) and both directions (en2twi, twi2en):
akanlm eval mt-select runs/cpt/akant5-small-m1x_en/final --study $STUDY --direction en2twi
akanlm eval mt-test   runs/cpt/akant5-small-m1x_en/final --study $STUDY --direction en2twi
# Seed means and paired bootstrap comparisons:
akanlm eval mt-report --study $STUDY
```

CPT checkpoints retain their trained `dropout_rate: 0.0`, while mT5, Afri-mT5-base and the
initializations store 0.1. The paper fine-tunes every model at 0.1. The explicit
`mt-freeze --dropout-rate 0.1` option records this fine-tuning setting in the frozen procedure and
applies it when constructing each model; it leaves the source checkpoint's weights and
`config.json` unchanged. Use the same option when registering more models in that study.
Without the option, the protocol retains the checkpoint's stored dropout rate for compatibility
with earlier experiments. The paper also decodes GhanaNLP Professional
(`configs/corpora/ghananlp_professional.yaml`) with the weights of each test run and the same
decoding settings; `mt-test --keep-weights` keeps those weights.

## Running on a cluster

Every step is a single `akanlm` command, so any scheduler can run it. Only CPT training needs a
distributed launch (`torchrun`, as above, with one process per GPU). The paper's runs used NVIDIA
B200 GPUs: two per CPT run for small and base and four for large (`configs/accel/b200_4gpu*.yaml`),
and one per MT fine-tuning job. Run the CPU-only preparation steps (`audit`, `tokenizer`,
`cpt --prepare-only`) as separate jobs, so that GPUs are not held idle.

## Data and licences

The code is released under the Apache License 2.0. The corpora are not redistributed; each
declaration in `configs/corpora/` names its Hub dataset, pinned revision and licence.

| Source | Role | Licence |
|---|---|---|
| Pristine Twi–English, GhOK Parallel, Ghana Farmer QA, True ASR IPA Pairs, Kumawood, Ghana Speech (Akuapem, Asante) | CPT | CC BY-NC 4.0 |
| Twi Wikipedia | CPT | CC BY-SA 3.0 |
| mC4 English (`allenai/c4`) | CPT (replay) and tokenizer training | ODC-BY |
| FLORES+ (Akuapem, Asante, English) | evaluation | CC BY-SA 4.0 |
| MAFAND-MT English–Twi | MT fine-tuning and evaluation | CC BY-NC 4.0 |
| GhanaNLP Professional | evaluation | see its dataset card |

### MAFAND-MT

The Hub repository `masakhane/mafand` contains only a loading script, and the auto-converted parquet
files that the loader reads (`hf_parquet_revision` in `configs/corpora/mafand_en_twi.yaml`) are no
longer served. The same data are in
[masakhane-io/lafand-mt](https://github.com/masakhane-io/lafand-mt) at commit
`c1a423500753a28b2fae8f5d5e1634de768cd89b`, whose `data/json_files/en-twi/{train,dev,test}.json`
are identical, pair by pair, to the files used in the paper (3,337, 1,284 and 1,500 pairs). Their
SHA-256 hashes are:

```
train.json  740ae25c22b5665e004e30e70366637967617d7bc6dfe6609be66ba7b15fb7a8
dev.json    e951fb42a976dce33e3645be182207200e0531eab89ceb754652398b72010eec
test.json   a5bf551c7f2ae90004b30be0365fef546f4a36a9f53dff4e221b63f0d3dd0c61
```

Write them as parquet where the loader looks for a local snapshot:

```python
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub.constants import HF_HUB_CACHE

revision = "ad2a4dff951cce6d7fd93566ecc527735ec1f68b"   # hf_parquet_revision in the declaration
root = Path(HF_HUB_CACHE) / "datasets--masakhane--mafand" / "snapshots" / revision / "en-twi"
for name, split in (("train", "train"), ("dev", "validation"), ("test", "test")):
    lines = Path(f"lafand-mt/data/json_files/en-twi/{name}.json").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    (root / split).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), root / split / "0000.parquet")
```

`akanlm eval mt-freeze` records a SHA-256 of the normalized items of every test set. For the paper's
data these are `abeb1ebc89ebd85dd1b4abb3ab9bd491ecd67f26acfd23a07d969117fdac4d89` (MAFAND-MT test),
`1582fba7ac2701c3501b26cdbcd6c1d9dea4c02373be45cf6783ea45909d8e3f` (FLORES+ Akuapem devtest) and
`60ca8c658183b775dcfbda780ee93d4f834f8fd57ccf6502c13df0b52aa966c9` (FLORES+ Asante devtest).

## Citation

See [`CITATION.cff`](CITATION.cff). The paper reference will be added on publication.
