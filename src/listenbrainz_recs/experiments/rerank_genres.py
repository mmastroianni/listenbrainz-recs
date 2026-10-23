"""Compare ALS nearest neighbors with MusicBrainz genre-aware reranking.

Two reranking methods are available:
  confidence  ALS score plus a genre adjustment scaled by tag-vote confidence (default)
  blend       linear interpolation of ALS and genre similarity
"""

import argparse
import sys

import numpy as np

from listenbrainz_recs.artists import load_artist_factors, resolve_artist, top_positions
from listenbrainz_recs.genres import (
    confidence_adjusted,
    genre_similarity,
    linear_blend,
    load_genres,
    neighborhood_coherence,
)


def print_coherence(name, pool, factors, genres):
    print(f"\n{name}: ALS neighborhood genre coherence")
    print(f"{'Top N':>6} {'Tagged':>8} {'Coverage':>10} {'Dispersion':>12} {'Pairwise':>10}")
    for n in (20, 50, 100, 200):
        positions = pool[:n]
        rows = [
            genres.rows[mbid]
            for mbid in (factors.mbid(pos) for pos in positions)
            if mbid in genres.rows
        ]
        coverage = f"{len(positions):6d} {len(rows):8d} {len(rows) / len(positions):10.1%}"
        result = neighborhood_coherence(genres, rows)
        if result is None:
            print(f"{coverage} {'--':>12} {'--':>10}")
        else:
            print(f"{coverage} {result['dispersion']:12.4f} {result['pairwise_similarity']:10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artist", nargs="?", default="Miles Davis")
    parser.add_argument("-n", "--number", type=int, default=20)
    parser.add_argument("--pool", type=int, default=500)
    parser.add_argument("--method", choices=("confidence", "blend"), default="confidence")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--neutral", type=float, default=0.3)
    parser.add_argument("--prior-votes", type=float, default=10.0)
    parser.add_argument(
        "--coherence", action="store_true", help="Print neighborhood genre coherence first"
    )
    parser.add_argument(
        "--coherence-only", action="store_true", help="Print coherence without rankings"
    )
    args = parser.parse_args()
    if args.number < 1 or args.pool < args.number:
        parser.error("Require 1 <= --number <= --pool")
    if any(not 0 <= w <= 1 for w in args.weights):
        parser.error("Weights must be between 0 and 1")
    if args.prior_votes <= 0:
        parser.error("--prior-votes must be positive")

    factors = load_artist_factors("GenreReranking")
    target_id = resolve_artist(args.artist, factors.artists)
    if target_id is None:
        sys.exit(1)
    target_pos = factors.position(target_id)
    if target_pos is None:
        sys.exit("Artist has no trained factor vector")

    als = factors.similarities(target_pos)
    pool = top_positions(als, min(args.pool, len(als) - 1))

    print("Loading genre vectors...", flush=True)
    genres = load_genres()
    target_mbid, target_name = factors.by_id[target_id]

    if args.coherence or args.coherence_only:
        print_coherence(target_name, pool, factors, genres)
        if args.coherence_only:
            return

    similarity, confidence = genre_similarity(
        genres, target_mbid, [factors.mbid(pos) for pos in pool], args.prior_votes
    )
    show_confidence = args.method == "confidence"

    def display(title, order, scores):
        print(f"\n{title}\n")
        conf_header = f" {'Conf':>6}" if show_confidence else ""
        print(f"{'#':>2}  {'Artist':40} {'ALS':>7} {'Genre':>7}{conf_header} {'Final':>7}")
        for rank, idx in enumerate(order[: args.number], 1):
            pos = pool[idx]
            genre = f"{similarity[idx]:.4f}" if np.isfinite(similarity[idx]) else "--"
            conf = f" {confidence[idx]:6.3f}" if show_confidence else ""
            print(
                f"{rank:2d}. {factors.label(pos)[:40]:40} "
                f"{als[pos]:7.4f} {genre:>7}{conf} {scores[idx]:7.4f}"
            )

    raw = als[pool]
    display(f"{target_name}: raw ALS", np.arange(len(pool)), raw)
    print(f"\nGenre coverage in candidate pool: {np.isfinite(similarity).sum()}/{len(pool)}")
    for weight in args.weights:
        if args.method == "confidence":
            scores = confidence_adjusted(raw, similarity, confidence, weight, args.neutral)
            title = f"{target_name}: genre adjustment {weight:.2f}"
        else:
            scores = linear_blend(raw, similarity, weight)
            title = f"{target_name}: genre weight {weight:.2f}"
        display(title, np.argsort(scores)[::-1], scores)


if __name__ == "__main__":
    main()
