"""Compare ALS cosine similarity with PCA-adjusted artist vectors."""

import argparse

import numpy as np
import pandas as pd

from listenbrainz_recs.paths import ARTIST_FACTORS, ARTIST_IDS, ARTIST_NAMES_CACHE, INTERACTIONS

DEFAULT_ARTISTS = [
    "Anthony Braxton",
    "Henry Threadgill",
    "Don Pullen",
    "Herbie Nichols",
    "Naked City",
    "Glenn Gould",
    "Taylor Swift",
]


def normalize(x):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def load_names():
    """Load or create a representative artist-name lookup."""
    if ARTIST_NAMES_CACHE.exists():
        print("Loading cached artist names...", flush=True)
        return pd.read_parquet(ARTIST_NAMES_CACHE)

    print(
        "Building artist-name cache from interactions...",
        flush=True,
    )

    interactions = pd.read_parquet(
        INTERACTIONS,
        columns=[
            "artist_mbid",
            "artist_name",
            "listen_count",
        ],
    )

    names = (
        interactions.dropna(subset=["artist_mbid", "artist_name"])
        .groupby(
            ["artist_mbid", "artist_name"],
            as_index=False,
        )["listen_count"]
        .sum()
        .sort_values("listen_count", ascending=False)
        .drop_duplicates("artist_mbid")[["artist_mbid", "artist_name"]]
    )

    ARTIST_NAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    names.to_parquet(ARTIST_NAMES_CACHE, index=False)

    print(
        f"Cached {len(names):,} artist names.",
        flush=True,
    )

    return names


def load_factors():
    """Load ALS factors and join them to MusicBrainz IDs."""
    factors = pd.read_parquet(ARTIST_FACTORS)
    ids = pd.read_parquet(ARTIST_IDS)

    print("Factor columns:", list(factors.columns))
    print("ID columns:", list(ids.columns))

    # Spark ALS itemFactors: id, features
    # Our mapping: artist_mbid, artist_id
    factors = factors.rename(columns={"id": "artist_id"})

    df = (
        factors.merge(
            ids,
            on="artist_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("artist_id")
        .reset_index(drop=True)
    )

    x = np.stack(df["features"].to_numpy()).astype(np.float32)

    names = load_names()

    df = df.merge(
        names,
        on="artist_mbid",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    assert len(df) == len(x), "Artist/factor alignment lost"

    return df, x


def compute_pca(x):
    """Compute PCA using the 64 x 64 covariance matrix."""
    mean = x.mean(axis=0, dtype=np.float64)
    centered = x.astype(np.float64) - mean

    covariance = (centered.T @ centered) / (len(x) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    ratios = eigenvalues / eigenvalues.sum()

    print("\nPCA explained variance:")
    for i, ratio in enumerate(ratios[:10], 1):
        print(f"  PC{i:2d}: {ratio:.2%}")

    print(f"  First 3 PCs combined: {ratios[:3].sum():.2%}")

    return (
        mean.astype(np.float32),
        eigenvectors.astype(np.float32),
    )


def make_representations(x, mean, components):
    """Build normalized vector representations."""
    centered = x - mean

    representations = {
        "raw": normalize(x),
        "centered": normalize(centered),
    }

    for k in (1, 2, 3):
        basis = components[:, :k]

        adjusted = centered - (centered @ basis) @ basis.T

        representations[f"remove_{k}"] = normalize(adjusted)

    # Subtract only half the projection onto PC1.
    pc1 = components[:, :1]

    partial = centered - 0.5 * (centered @ pc1) @ pc1.T

    representations["half_pc1"] = normalize(partial)

    return representations


def find_artist(df, name):
    """Find an artist by exact case-insensitive name."""
    names = df["artist_name"].fillna("")

    exact = np.flatnonzero(names.str.casefold().to_numpy() == name.casefold())

    if len(exact) == 1:
        return int(exact[0])

    if len(exact) > 1:
        print(f"\nMultiple exact matches for {name}:")
        for i in exact[:10]:
            print(f"  {df.iloc[i]['artist_name']} ({df.iloc[i]['artist_mbid']})")
        return None

    partial = np.flatnonzero(
        names.str.contains(
            name,
            case=False,
            regex=False,
        ).to_numpy()
    )

    if len(partial):
        print(f"\nNo exact match for {name}. Possible matches:")
        for i in partial[:15]:
            print(f"  {df.iloc[i]['artist_name']}")
    else:
        print(f"\nArtist not found: {name}")

    return None


def top_neighbors(matrix, index, n):
    """Find top-n neighbors without an all-pairs matrix."""
    scores = matrix @ matrix[index]
    scores[index] = -np.inf

    n = min(n, len(scores) - 1)

    candidates = np.argpartition(scores, -n)[-n:]

    ordered = candidates[np.argsort(scores[candidates])[::-1]]

    return ordered, scores


def compare_artist(df, representations, artist, n):
    """Print nearest neighbors for each representation."""
    index = find_artist(df, artist)

    if index is None:
        return

    print(f"\n{'#' * 72}")
    print(f"ARTIST: {artist}")
    print(f"{'#' * 72}")

    for method, matrix in representations.items():
        neighbors, scores = top_neighbors(matrix, index, n)

        print(f"\n--- {method.upper()} ---")

        for rank, j in enumerate(neighbors, 1):
            name = df.iloc[j]["artist_name"]

            if pd.isna(name):
                name = df.iloc[j]["artist_mbid"]

            print(f"{rank:2d}. {str(name):42.42s} {scores[j]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=("Compare raw ALS cosine similarity with centered and PCA-adjusted vectors.")
    )

    parser.add_argument(
        "artists",
        nargs="*",
        help="Artist names (default: seven test artists)",
    )

    parser.add_argument(
        "-n",
        "--neighbors",
        type=int,
        default=15,
        help="Number of neighbors (default: 15)",
    )

    args = parser.parse_args()

    if args.neighbors < 1:
        parser.error("--neighbors must be positive")

    artists = args.artists or DEFAULT_ARTISTS

    print("Loading ALS factors...", flush=True)
    df, x = load_factors()

    print(f"Loaded {len(df):,} artists, {x.shape[1]} dimensions")

    print(f"Factor matrix: {x.nbytes / 1024**2:.1f} MiB")

    print("Computing PCA...", flush=True)
    mean, components = compute_pca(x)

    print(
        "Creating adjusted representations...",
        flush=True,
    )

    representations = make_representations(x, mean, components)

    for artist in artists:
        compare_artist(
            df,
            representations,
            artist,
            args.neighbors,
        )


if __name__ == "__main__":
    main()
