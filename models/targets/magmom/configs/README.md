# configs

One YAML per backend and graph cutoff. Every backend shares the same architecture, an atomic-number embedding, three message-passing layers, and a per-node readout MLP that outputs one magnetic moment per atom (the file stem `c128l3h1` means conv width 128, 3 conv layers, 1 readout hidden layer). Only the message-passing layer differs between backends, so a head-to-head isolates the operator. Each backend has three cutoffs, `r4`, `r6`, `r8` (neighbour radius in Angstrom), with `r6` a sensible default.

Run one backend at one cutoff with:

```
./scripts/02-submit-training.sh --target magmom --dataset magmom21-v0p1 \
    targets/magmom/configs/cgconv/c128l3h1-r6.yaml
```

Or submit a selection of backends in one command (trains all three cutoffs by default, run with `-h` to list the available backends):

```
./scripts/submit-selection.sh cgconv schnetconv gatv2conv
```

## Node-level backends

Native per-atom graph-convolution layers, the main candidates. Several consume the scalar interatomic distance as an edge feature (marked "uses distance").

- [CGConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CGConv.html): crystal-graph convolution (Xie and Grossman 2018) built for periodic materials, uses distance. A natural fit for metal slabs.
- [SchNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.SchNet.html): continuous-filter convolution with radial-basis-expanded distances (Schuett et al. 2017), a common molecules-and-materials baseline, uses distance.
- [GraphConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GraphConv.html): weighted neighbour-sum convolution (Morris et al. 2019), no edge features, a simple strong baseline.
- [SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html): GraphSAGE mean-aggregation convolution (Hamilton et al. 2017), no edge features.
- [GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html): graph attention with dynamic attention scores (Brody et al. 2021), learns a weight per neighbour, uses distance.
- [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html): classic spectral graph convolution (Kipf and Welling 2017), symmetric-normalised neighbour averaging, no edge features.
- [TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html): graph-transformer multi-head attention (Shi et al. 2020), uses distance.
- [GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html): graph isomorphism network with edge features (Hu et al. 2019), high expressive power, uses distance.
- [NNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.NNConv.html): edge-conditioned convolution (Gilmer et al. 2017), a small network turns each distance into a message weight matrix, uses distance.
- [GENConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GENConv.html): DeeperGCN generalised aggregation (Li et al. 2020), softmax or power-mean message pooling.
- [GMMConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GMMConv.html): MoNet Gaussian-mixture convolution (Monti et al. 2017) over distance pseudo-coordinates, uses distance.
- [ResGatedGraphConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ResGatedGraphConv.html): residual gated graph convolution (Bresson and Laurent 2017), edge gates modulate messages, uses distance.
- [GeneralConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GeneralConv.html): configurable design-space convolution (You et al. 2020), uses distance.
- [PDNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PDNConv.html): pathfinder discovery network convolution (Rozemberczki et al. 2021), edge features modulate message passing, uses distance.
- [SplineConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SplineConv.html): SplineCNN B-spline-kernel convolution (Fey et al. 2018) over distance pseudo-coordinates, uses distance, needs the `torch-spline-conv` package.

## Reference backends

Heavier, slower, higher-accuracy references rather than deployment candidates, adapted to per-atom output. All three are periodic (PBC-aware), so they get the slab's true in-plane neighbours. Validate on the first run with `scratch/preflight-pbc.py`.

- [DimeNet++](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DimeNetPlusPlus.html): directional message passing over atom triplets and bond angles (Gasteiger et al. 2020). PyG model, made PBC-aware here by feeding it the periodic graph and minimum-image displacement vectors (OCP-style).
- [ET](https://torchmd-net.readthedocs.io/): torchmd-net Equivariant Transformer (Tholke and de Fabritiis 2022), scalar-vector equivariant, native PBC. The architecture ViSNet extends.
- [TensorNet](https://torchmd-net.readthedocs.io/): torchmd-net rank-2 Cartesian-tensor equivariant network (Simeon and de Fabritiis 2023), native PBC. Needs `torchmd-net` installed.
