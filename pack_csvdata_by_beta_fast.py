#!/usr/bin/env python

import argparse
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


CHANNELS = ["Bx", "By", "Bz", "Density"]

OFFICIAL_BETAS = ["0.01", "0.02", "0.04", "0.06", "0.08", "0.1", "0.12", "0.15", "0.18", "0.2"]
OFFICIAL_NUS   = {"0", "1", "2", "3", "5", "6", "7", "8", "9", "10"}
OFFICIAL_BZS   = {"0", "0.1", "0.15", "0.2", "0.3"}
OFFICIAL_DTS   = {"2"}
OFFICIAL_TAUS  = {"40", "70", "100", "150", "200"}

RUN_RE = re.compile(
    r"^beta(?P<beta>[0-9.]+)_nu(?P<nu>[0-9]+)_Bz(?P<Bz>[0-9.]+)_dt(?P<dt>[0-9]+)_tau(?P<tau>[0-9]+)$"
)

FRAME_RE = re.compile(r"_(\d+)\.csv$")


def parse_run_name(run_name):
    m = RUN_RE.match(run_name)
    if m is None:
        return None
    return m.groupdict()


def is_official_run(params):
    return (
        params["beta"] in set(OFFICIAL_BETAS)
        and params["nu"] in OFFICIAL_NUS
        and params["Bz"] in OFFICIAL_BZS
        and params["dt"] in OFFICIAL_DTS
        and params["tau"] in OFFICIAL_TAUS
    )


def get_frame_id(path):
    m = FRAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse frame id from {path}")
    return int(m.group(1))


def read_csv_float32(path):
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def read_frame_batch(task):
    """
    Worker function.

    Returns:
        start_index, block
        block shape = (n_frames_in_batch, 4, Nx, Nz)
    """
    start_index, frame_items, expected_shape = task
    Nx, Nz = expected_shape

    block = np.empty((len(frame_items), len(CHANNELS), Nx, Nz), dtype=np.float32)

    for local_i, item in enumerate(frame_items):
        # item: (frame_id, [Bx_path, By_path, Bz_path, Density_path])
        fid, paths = item

        for c, path in enumerate(paths):
            arr = read_csv_float32(path)

            if arr.shape != (Nx, Nz):
                raise RuntimeError(
                    f"Shape mismatch in {path}: expected {(Nx, Nz)}, got {arr.shape}"
                )

            block[local_i, c] = arr

    return start_index, block


def get_var_files(csv_dir, var):
    files = sorted(csv_dir.glob(f"{var}_*.csv"), key=get_frame_id)
    ids = np.array([get_frame_id(f) for f in files], dtype=np.int32)
    file_map = {int(fid): str(f) for fid, f in zip(ids, files)}
    return ids, file_map


def find_official_runs_for_beta(runs_dir, beta_value):
    run_dirs = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        params = parse_run_name(run_dir.name)
        if params is None:
            continue

        if not is_official_run(params):
            continue

        if params["beta"] != beta_value:
            continue

        if not (run_dir / "CSV_Data").exists():
            continue

        run_dirs.append(run_dir)

    return run_dirs


def get_compression_kwargs(compression):
    if compression == "none":
        return {}

    if compression == "lzf":
        return {
            "compression": "lzf",
            "shuffle": True,
        }

    if compression == "gzip":
        return {
            "compression": "gzip",
            "compression_opts": 4,
            "shuffle": True,
        }

    raise ValueError(f"Unknown compression: {compression}")


def make_batches(frame_items, batch_frames):
    for start in range(0, len(frame_items), batch_frames):
        yield start, frame_items[start:start + batch_frames]


def pack_one_run(h5, run_dir, compression_kwargs, pool, batch_frames=4, overwrite=False):
    run_name = run_dir.name
    csv_dir = run_dir / "CSV_Data"

    params_str = parse_run_name(run_name)
    if params_str is None:
        raise ValueError(f"Bad run name: {run_name}")

    runs_group = h5.require_group("runs")

    if run_name in runs_group:
        completed = bool(runs_group[run_name].attrs.get("completed", False))
        if completed and not overwrite:
            return "skipped"
        del runs_group[run_name]

    ids_by_var = {}
    files_by_var = {}

    for var in CHANNELS:
        ids, file_map = get_var_files(csv_dir, var)

        if len(ids) == 0:
            raise RuntimeError(f"No {var}_*.csv found in {csv_dir}")

        ids_by_var[var] = ids
        files_by_var[var] = file_map

    frame_ids = ids_by_var["Bx"]

    for var in ["By", "Bz", "Density"]:
        if not np.array_equal(ids_by_var[var], frame_ids):
            raise RuntimeError(f"Frame ID mismatch in {run_name}: Bx vs {var}")

    if frame_ids[0] != 0:
        raise RuntimeError(f"{run_name}: first frame ID is not 0")

    if not np.array_equal(frame_ids, np.arange(frame_ids[-1] + 1)):
        raise RuntimeError(f"{run_name}: frame IDs are not continuous")

    T = len(frame_ids)

    sample = read_csv_float32(files_by_var["Density"][int(frame_ids[0])])
    Nx, Nz = sample.shape

    g = runs_group.create_group(run_name)
    g.attrs["completed"] = False

    g.attrs["run_name"] = run_name
    g.attrs["channel_order"] = ",".join(CHANNELS)
    g.attrs["field_shape"] = "(T, 4, Nx, Nz)"

    g.attrs["beta"] = float(params_str["beta"])
    g.attrs["nu"] = int(params_str["nu"])
    g.attrs["Bz0"] = float(params_str["Bz"])
    g.attrs["dt"] = int(params_str["dt"])
    g.attrs["tau"] = int(params_str["tau"])

    g.attrs["T"] = T
    g.attrs["Nx"] = Nx
    g.attrs["Nz"] = Nz

    g.create_dataset(
        "frame_ids",
        data=frame_ids,
        dtype=np.int32,
    )

    fields = g.create_dataset(
        "fields",
        shape=(T, len(CHANNELS), Nx, Nz),
        dtype=np.float32,
        chunks=(1, len(CHANNELS), Nx, Nz),
        **compression_kwargs,
    )

    frame_items = []

    for fid in frame_ids:
        fid = int(fid)
        paths = [
            files_by_var["Bx"][fid],
            files_by_var["By"][fid],
            files_by_var["Bz"][fid],
            files_by_var["Density"][fid],
        ]
        frame_items.append((fid, paths))

    tasks = [
        (start, batch, (Nx, Nz))
        for start, batch in make_batches(frame_items, batch_frames)
    ]

    futures = [pool.submit(read_frame_batch, task) for task in tasks]

    for fut in as_completed(futures):
        start_index, block = fut.result()
        end_index = start_index + block.shape[0]
        fields[start_index:end_index] = block

    g.attrs["completed"] = True
    return "written"


def build_index(h5):
    if "index" in h5:
        del h5["index"]

    idx = h5.create_group("index")
    runs_group = h5["runs"]

    run_names = []
    beta = []
    nu = []
    Bz0 = []
    dt = []
    tau = []
    T = []
    Nx = []
    Nz = []

    for rn in sorted(runs_group.keys()):
        g = runs_group[rn]

        if not bool(g.attrs.get("completed", False)):
            continue

        run_names.append(rn)
        beta.append(float(g.attrs["beta"]))
        nu.append(int(g.attrs["nu"]))
        Bz0.append(float(g.attrs["Bz0"]))
        dt.append(int(g.attrs["dt"]))
        tau.append(int(g.attrs["tau"]))
        T.append(int(g.attrs["T"]))
        Nx.append(int(g.attrs["Nx"]))
        Nz.append(int(g.attrs["Nz"]))

    str_dtype = h5py.string_dtype(encoding="utf-8")

    idx.create_dataset("run_names", data=run_names, dtype=str_dtype)
    idx.create_dataset("beta", data=np.array(beta, dtype=np.float32))
    idx.create_dataset("nu", data=np.array(nu, dtype=np.int32))
    idx.create_dataset("Bz0", data=np.array(Bz0, dtype=np.float32))
    idx.create_dataset("dt", data=np.array(dt, dtype=np.int32))
    idx.create_dataset("tau", data=np.array(tau, dtype=np.int32))
    idx.create_dataset("T", data=np.array(T, dtype=np.int32))
    idx.create_dataset("Nx", data=np.array(Nx, dtype=np.int32))
    idx.create_dataset("Nz", data=np.array(Nz, dtype=np.int32))

    h5.attrs["num_runs"] = len(run_names)
    h5.attrs["channel_order"] = ",".join(CHANNELS)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs-dir",
        type=str,
        default="/pscratch/sd/d/dcfy/VPIC_PPPL/runs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta",
    )
    parser.add_argument(
        "--beta",
        type=str,
        required=True,
        choices=OFFICIAL_BETAS,
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["none", "lzf", "gzip"],
        default="lzf",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="For testing only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beta_tag = args.beta
    out = out_dir / f"VPIC_PPPL_CSV_Data_official2500_beta{beta_tag}_{args.compression}.h5"

    run_dirs = find_official_runs_for_beta(runs_dir, beta_tag)

    print(f"beta = {beta_tag}")
    print(f"Found official runs for beta={beta_tag}: {len(run_dirs)}")

    if len(run_dirs) != 250:
        raise RuntimeError(f"Expected 250 runs for beta={beta_tag}, found {len(run_dirs)}")

    if args.max_runs is not None:
        run_dirs = run_dirs[:args.max_runs]
        print(f"Testing mode: using only {len(run_dirs)} runs")

    print(f"Output: {out}")
    print(f"Compression: {args.compression}")
    print(f"Workers: {args.workers}")
    print(f"Batch frames: {args.batch_frames}")
    print(f"Channels: {CHANNELS}")

    compression_kwargs = get_compression_kwargs(args.compression)

    bad_runs = []

    # spawn 比 fork 更安全，避免 HDF5 handle 被 worker 继承
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        with h5py.File(out, "a") as h5:
            h5.attrs["description"] = f"Official VPIC PPPL runs for beta={beta_tag}; CSV_Data Bx, By, Bz, Density only"
            h5.attrs["source_runs_dir"] = str(runs_dir)
            h5.attrs["field_shape"] = "(T, 4, Nx, Nz)"
            h5.attrs["channel_order"] = ",".join(CHANNELS)
            h5.attrs["official_only"] = True
            h5.attrs["beta_file"] = float(beta_tag)

            for run_dir in tqdm(run_dirs):
                try:
                    pack_one_run(
                        h5,
                        run_dir,
                        compression_kwargs=compression_kwargs,
                        pool=pool,
                        batch_frames=args.batch_frames,
                        overwrite=args.overwrite,
                    )
                    h5.flush()

                except Exception as e:
                    bad_runs.append((run_dir.name, repr(e)))
                    print(f"\n[BAD RUN] {run_dir.name}: {e}")

                    if "runs" in h5 and run_dir.name in h5["runs"]:
                        del h5["runs"][run_dir.name]
                    h5.flush()

            build_index(h5)
            h5.flush()

    if bad_runs:
        bad_log = out.with_suffix(".bad_runs.txt")
        with open(bad_log, "w") as f:
            for rn, err in bad_runs:
                f.write(f"{rn}\t{err}\n")
        print(f"Bad runs saved to {bad_log}")

    print("Done.")


if __name__ == "__main__":
    main()
