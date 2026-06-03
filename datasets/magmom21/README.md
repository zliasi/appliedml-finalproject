# magmom21

Per-atom magnetic moments of metal alloy surface slabs (ordered and random), computed with DFT in GPAW.

## Elements

21 metals across FCC, BCC, and HCP: Ag, Al, Au, Co, Cr, Cu, Fe, Ir, Mn, Mo, Ni, Os, Pd, Pt, Re, Rh, Ru, Ti, V, W, Zn. The magnetic ones, which trigger spin polarization, are Co, Cr, Cu, Fe, Mn, Mo, Ni, Os, Ru, V, W.

## Slabs

- Lattices: FCC(111), BCC(110), HCP(0001), picked per composition by a weighted vote over the constituent elements (per element choice is based on EOS results).
- Size: 4x4x4 (64 atoms) for random alloys + dilute structures, 2x2x4 for binary intermetallic prototypes, and 4x4x4 for ternary Heuslers.
- Bottom 2 layers fixed. 10 Å vacuum. periodic in-plane (pbc [True, True, False]).
- Two families: random alloys (`data/compositions.csv`) and ordered/dilute placements (`data/ordered-compositions.csv`).

## Computation

GPAW in plane-wave mode (350 eV cutoff), RPBE functional, Monkhorst-Pack k-points at 3.5 1/Å density, Marzari-Vanderbilt smearing, dipole correction along the surface normal. Calculations are spin-polarized whenever a magnetic element is present, with initial moments seeded from per-element, per-lattice equation-of-state values.

Per slab: a UMA pre-relaxation (UMA-S-1.2), then a staged GPAW run: (i) an LCAO warm-up using sz(dzp) basis, (ii) followed by plane-wave geometry relaxation (LBFGS, fmax 0.2 eV/Å). The SCF calculations used a 4-level retry ladder (Pulay/MSR1 density mixing for stages 0-2, direct minimisation for stage 3) that progressively loosens the eigenstate criterion (1e-8 to 1e-5) and widens the smearing. Per-atom moments are read from the converged density.

## Files

- `data/magmoms.h5`: per slab, atom-ordered: `magmoms` (target), `atomic_numbers`, `positions`, `cell`, `tags`, `pbc`, plus energy/work_function/fermi_level/spinpol attributes.
- `data/compositions.csv`, `data/ordered-compositions.csv`: composition table for the train/val/test split.

