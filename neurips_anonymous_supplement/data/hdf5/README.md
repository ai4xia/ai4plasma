# Processed VPIC HDF5 layout

The processed VPIC corpus used in the paper is not distributed with this
anonymous submission.

When the corpus is supplied locally, this directory (or `--h5-dir` /
`H5_DIR`) should contain files named:

```text
processed_vpic_beta0.2.h5
```

Expected layout:

- HDF5 field array shape `(T, 4, X, Z)`
- channels `Bx, By, Bz, Density`
- the paper uses beta = 0.2
- 250 runs total
- 225 train / 25 held-out
- the exact split is in `configs/split.json`
- training windows are 24 frames with temporal stride 2
- paper spatial size is `X=154`, `Z=62`
- physical plot extent `[zmin, zmax, xmin, xmax] = [-21, 21, -50, 50]` cm

Do not assume a public download URL exists.
