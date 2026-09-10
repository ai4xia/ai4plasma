#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
H5_DIR="${H5_DIR:-${ROOT}/data/hdf5}"
if ! ls "${H5_DIR}"/processed_vpic_beta*.h5 >/dev/null 2>&1; then
    echo "The processed VPIC corpus used in the paper is not distributed with this anonymous submission."
    echo "Place the HDF5 files in data/hdf5 or set H5_DIR / --h5-dir to that directory."
    exit 2
fi

python make_paper_figures.py \
    --run-dir "${ROOT}" \
    --checkpoint checkpoint/latest.pt \
    --h5-dir "${H5_DIR}" \
    --sliding-max-runs 25

python evaluate_standardized_validation.py \
    --run-dir "${ROOT}" \
    --checkpoint checkpoint/latest.pt \
    --h5-dir "${H5_DIR}"
