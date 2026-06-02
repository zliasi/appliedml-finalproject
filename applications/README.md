# applications

Example application of the atomic magnetic moments GNN. A trained checkpoint is used to predict atomic magnetic moments for a given slab, which are then fed to a GPAW calculator, so the SCF starts from a reasonable magnetic state.

- `infer-magmoments.py` - `predict_magmoms(atoms, checkpoint) -> array`
- `relax-slabs.py` - create a slab, predict, set moments, run a short relaxation

