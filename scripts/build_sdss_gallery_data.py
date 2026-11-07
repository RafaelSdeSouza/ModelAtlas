"""Rebuild the two public SDSS gallery subsets from pinned astroML FITS files."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

FILES = {
    "SDSSspecgalsDR8_1.fit.gz": "32bdd24ceaf24bc1090af8eb46f37bd2419d0e2e3a694f4759c177969c5e0bc8",
    "SDSSspecgalsDR8_2.fit.gz": "b7b9cee8c159fb54bcf70871895c0f653dc2293f5f70b29726224d2ce2b482bc",
    "SDSSssppDR9_rerun122.fit.gz": "e305979ff26f818fdc5d4959f728b2a3b63afe9d506e9b551a0319d9d568bdf4",
}
SAMPLE_SEED = 20261107


def read_catalogue(path, columns):
    from astropy.io import fits

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != FILES[path.name]:
        raise ValueError(f"Upstream catalogue checksum differs for {path.name}")
    with fits.open(path, memmap=False) as hdus:
        return {name: hdus[1].data[name].copy() for name in columns}


def sample_rows(rows, size):
    if len(rows) < size:
        raise ValueError(f"Expected at least {size} selected records, found {len(rows)}")
    return np.sort(np.random.default_rng(SAMPLE_SEED).choice(rows, size, replace=False))


def galaxy_subset(source, output):
    bands = "ugriz"
    mags = [f"modelMag_{band}" for band in bands]
    errors = [f"modelMagErr_{band}" for band in bands]
    columns = ["objID", "plate", "z", "zErr", "extinction_r", "petroMag_r", *mags, *errors]
    halves = [read_catalogue(source / name, columns) for name in list(FILES)[:2]]
    data = {name: np.concatenate([half[name] for half in halves]) for name in columns}
    magnitude = np.column_stack([data[name] for name in mags]).astype(np.float64)
    mag_error = np.column_stack([data[name] for name in errors])
    valid = (
        np.all(np.isfinite(magnitude), axis=1)
        & np.all((mag_error > 0) & (mag_error <= [0.15, 0.08, 0.08, 0.08, 0.08]), axis=1)
        & (data["petroMag_r"] >= 14)
        & (data["petroMag_r"] <= 17.77)
        & (data["extinction_r"] >= 0)
        & (data["extinction_r"] <= 0.15)
        & (data["z"] >= 0.02)
        & (data["z"] <= 0.35)
        & (data["zErr"] > 0)
        & (data["zErr"] <= 0.001)
    )
    rows = np.flatnonzero(valid)
    # Among repeated imaging IDs keep the smallest pipeline redshift error.
    order = np.lexsort((rows, data["zErr"][rows], data["objID"][rows]))
    by_quality = rows[order]
    _, first = np.unique(data["objID"][by_quality], return_index=True)
    rows = sample_rows(by_quality[first], 12_000)
    np.savez_compressed(
        output / "sdss_dr8_galaxies.npz",
        colors=-np.diff(magnitude[rows], axis=1).astype("<f4"),
        redshift=data["z"][rows].astype("<f4"),
        plate=data["plate"][rows].astype("<i2"),
        source_row=rows.astype("<i4"),
    )
    print(f"SDSS DR8 galaxies: {len(first):,} eligible unique objects, {len(rows):,} saved.")


def stellar_subset(source, output):
    columns = (
        "ra",
        "dec",
        "Ar",
        "upsf",
        "gpsf",
        "rpsf",
        "uErr",
        "gErr",
        "rErr",
        "plate",
        "SNR",
        "FeH",
        "FeHErr",
        "Teff",
        "logg",
    )
    data = read_catalogue(source / "SDSSssppDR9_rerun122.fit.gz", columns)
    extinction_r = data["Ar"].astype(np.float64)
    u = data["upsf"] - 1.810 * extinction_r
    g = data["gpsf"] - 1.400 * extinction_r
    r = data["rpsf"] - extinction_r
    colors = np.column_stack((u - g, g - r))
    valid = (
        np.all(np.isfinite(colors), axis=1)
        & (extinction_r >= 0)
        & (extinction_r <= 0.20)
        & (g >= 15)
        & (g < 19.5)
        & (colors[:, 0] > 0)
        & (colors[:, 0] < 3)
        & (colors[:, 1] >= 0.20)
        & (colors[:, 1] <= 0.60)
        & (data["logg"] >= 4)
        & (data["Teff"] >= 5000)
        & (data["Teff"] <= 7000)
        & (data["FeH"] >= -3)
        & (data["FeH"] <= 0.5)
        & (data["FeHErr"] > 0)
        & (data["FeHErr"] <= 0.20)
        & (data["SNR"] >= 20)
        & (data["uErr"] > 0)
        & (data["uErr"] <= 0.10)
        & (data["gErr"] > 0)
        & (data["gErr"] <= 0.05)
        & (data["rErr"] > 0)
        & (data["rErr"] <= 0.05)
    )
    rows = np.flatnonzero(valid)
    ra, dec = np.deg2rad([data["ra"][rows], data["dec"][rows]])
    unit = np.column_stack((np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)))
    radius = 2 * np.sin(np.deg2rad(1 / 3600) / 2)
    left, right = cKDTree(unit).query_pairs(radius, output_type="ndarray").T
    snr = data["SNR"][rows]
    remove = np.where(snr[left] >= snr[right], right, left)
    keep = np.ones(len(rows), dtype=bool)
    keep[remove] = False
    rows = sample_rows(rows[keep], 8_000)
    np.savez_compressed(
        output / "sdss_dr9_stars.npz",
        colors=colors[rows].astype("<f4"),
        teff=data["Teff"][rows].astype("<f4"),
        feh=data["FeH"][rows].astype("<f4"),
        plate=data["plate"][rows].astype("<i2"),
        source_row=rows.astype("<i4"),
    )
    print(f"SDSS DR9 stars: {keep.sum():,} eligible distinct records, {len(rows):,} saved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalogue_dir", type=Path, help="Directory with the three pinned FITS files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "astronomy",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    galaxy_subset(args.catalogue_dir, args.output_dir)
    stellar_subset(args.catalogue_dir, args.output_dir)


if __name__ == "__main__":
    main()
