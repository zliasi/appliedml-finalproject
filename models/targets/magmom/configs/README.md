# configs

One yaml per node-level backend and cutoff (r4/r6/r8): cgconv, schnetconv, graphconv, sageconv, gatv2conv, gcnconv, transformerconv, gineconv, nnconv, genconv, gmmconv, resgatedgraphconv, generalconv, pdnconv, splineconv. dimenet and visnet are the slow, accurate reference backends (PyG DimeNet++ / ViSNet wrapped and adapted to per-atom output; validate on first run). pnaconv needs the train-set degree histogram at build time.
