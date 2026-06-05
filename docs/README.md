# Getting started: train magmom GNN backends

A step-by-step guide to train a few GNN models on the shared magmom21 dataset. Pre-made graphs are available on Steno. Run everything on the HPC login node.

All Slurm jobs use a shared conda environment automatically, so there is nothing to set up, when running on Steno.

## 1. Clone the repository

The repository is public, so no login is needed. From the directory where you want the project to live:

```
git clone https://github.com/zliasi/appliedml-finalproject.git
```

This creates `~/appliedml-finalproject` with all the code.

## 2. Link the dataset graphs

The GNN training reads prebuilt graph files. They are large (about 1-3 GB) so they are not in the repository, they live in a shared folder you can read. Link to them instead of copying:

```
cd ~/appliedml-finalproject/models
mkdir -p runs/magmom-magmom21-v0p1
ln -s /groups/kemi/liasi/share/magmom21-graphs runs/magmom-magmom21-v0p1/graphs
```

You should see nine `.lmdb` files:

```
ls runs/magmom-magmom21-v0p1/graphs/
```

## 3. Pick your backends and submit training

The backends are the different GNN layer types. List them with a one-line description of each:

```
./scripts/submit-selection.sh -h
```

For longer descriptions, read `targets/magmom/configs/README.md`.

Pick four or five backends and submit, for example:

```
./scripts/submit-selection.sh cgconv sageconv gatv2conv gineconv transformerconv
```

Just replace the backend names above with your picks. This trains each of your backends at three graph cutoffs (r4, r6, r8), so three GPU jobs are submitted per backend.

## Optional: track training with Weights and Biases

If you want live training curves (loss per epoch and so on), you can log to your own Weights and Biases account.

Create a free account at https://wandb.ai, then once on Steno run:

```
wandb login
```

Paste your API key from https://wandb.ai/authorize when asked. It is saved to your home directory, so the jobs can use it.

Then add `--wandb` when you submit:

```
./scripts/submit-selection.sh --wandb cgconv sageconv gatv2conv gineconv transformerconv
```

Your runs appear under your own account in a project called `magmom-magmom21-v0p1`. This does not change the trained models or the results.

## 4. Check on your jobs

Each job writes a log file to `runs/magmom-magmom21-v0p1/logs/` and, when it finishes, a trained model to `runs/magmom-magmom21-v0p1/checkpoints/`.

When all your jobs have finished, evaluate your models (step 5) and upload the results (step 6).

## 5. Evaluate your models

This produces a parity figure, an error figure, and a metrics file for each trained model. From the `models` directory:

```
source scripts/env.sh
./scripts/04-submit-evaluate.sh --target magmom --dataset magmom21-v0p1
```

It runs on the login node and takes a few minutes. The figures and metrics are written to `runs/magmom-magmom21-v0p1/eval/`.

## 6. Upload your results to the repo

Your results are the trained models in `runs/magmom-magmom21-v0p1/checkpoints/` and the figures and metrics in `runs/magmom-magmom21-v0p1/eval/`. Push both to your own branch so we can compare everyone's models at the end.

First time only, connect your GitHub account to Steno with an SSH key:

```
ssh-keygen -t ed25519 -C "your-github-email"
cat ~/.ssh/id_ed25519.pub
```

Press Enter through the `ssh-keygen` prompts. Copy the whole line that `cat` prints, then on github.com go to Settings, then SSH and GPG keys, then New SSH key, paste it, and save. Then point your clone at GitHub over SSH. From the appliedml-finalproject directory run:

```
git remote set-url origin git@github.com:zliasi/appliedml-finalproject.git
```

Then upload your results, replacing `yourname` with your own name:

```
git checkout -b results-yourname
git add models/runs/magmom-magmom21-v0p1/checkpoints models/runs/magmom-magmom21-v0p1/eval
git commit -m "results-yourname"
git push -u origin results-yourname
```

This puts your trained models, figures, and metrics on a branch called `results-yourname`. Then you're done.
