# GNN tutorial

A self-contained, beginner-friendly tutorial on training a graph neural network (GNN) on atomic structures. The notebook trains a GNN to predict the work function of slabs from their atomic arrangement.

## Contents

- `train-a-gnn.ipynb` - the tutorial, fully worked. Work through it top to bottom.
- `dataset/` - everything the notebook needs:
  - `work-functions.json` - the labels (work function per structure).
  - `slabs/<composition>/<variant>.traj` - the atomic structures (ASE format).

The data is a four-metal composition space sample (Ag, Au, Pd, Pt): 199 compositions, about 870 structures. The dataset is small enough that CPU can be used throughout. For the alloy slabs, there are a number of unique variants per composition, to give the model a proper sample of that data point.

## Running it

The notebook is self-contained: its first cell ("Dependencies") installs the packages it needs and downloads the dataset. You do not need to install anything or clone the repo by hand.

### Jupyter

Open `train-a-gnn.ipynb` in Jupyter (JupyterLab or the classic notebook) and run the cells top to bottom. Starting from scratch:

```
git clone https://github.com/zliasi/appliedml-finalproject.git
cd appliedml-finalproject/tutorial
jupyter lab train-a-gnn.ipynb
```

The first cell installs any missing packages (PyTorch, PyTorch Geometric, ASE, networkx, matplotlib), or you can install them yourself with `pip install -r ../requirements.txt`. Everything runs on CPU in a couple of minutes

### Google Colab

Open the notebook in Colab and run the first cell:

https://colab.research.google.com/github/zliasi/appliedml-finalproject/blob/main/tutorial/train-a-gnn.ipynb

Colab downloads the dataset from the repository.
