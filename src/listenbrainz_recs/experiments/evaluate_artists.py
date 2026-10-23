"""Blinded, graded relevance evaluation of raw ALS vs confidence-weighted genre reranking.

export  write a blinded judgments CSV (fill in relevance 0-3) plus a separate details CSV
score   compute NDCG@k for both rankings from the completed judgments
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from listenbrainz_recs.artists import load_artist_factors, resolve_artist, top_positions
from listenbrainz_recs.genres import confidence_adjusted, genre_similarity, load_genres
from listenbrainz_recs.paths import RESULTS_DIR

FIELDS = ["seed_mbid", "seed_artist", "candidate_mbid", "candidate_artist", "relevance"]
DETAILS = ["als_score", "genre_score", "confidence", "raw_rank", "hybrid_rank", "hybrid_score"]
DEFAULT_JUDGMENTS = RESULTS_DIR / "artist_judgments.csv"
DEFAULT_DETAILS = RESULTS_DIR / "artist_ranking_details.csv"


def export(args):
    factors = load_artist_factors("ArtistEvaluation")
    genres = load_genres()
    rows = []
    details = []
    for seed in args.artists:
        target_id = resolve_artist(seed, factors.artists)
        if target_id is None:
            sys.exit(f"Could not resolve seed: {seed}")
        target_pos = factors.position(target_id)
        if target_pos is None:
            sys.exit(f"No factor for seed: {seed}")
        scores = factors.similarities(target_pos)
        pool = top_positions(scores, min(args.pool, len(scores) - 1))
        raw = scores[pool]
        seed_mbid, seed_name = factors.by_id[target_id]
        genre_sim, conf = genre_similarity(
            genres, seed_mbid, [factors.mbid(pos) for pos in pool], args.prior_votes
        )
        hybrid = confidence_adjusted(raw, genre_sim, conf, args.weight, args.neutral)

        raw_order = np.argsort(-raw, kind="stable")
        hybrid_order = np.argsort(-hybrid, kind="stable")
        raw_ranks = np.empty(len(pool), dtype=int)
        hybrid_ranks = np.empty(len(pool), dtype=int)
        raw_ranks[raw_order] = np.arange(1, len(pool) + 1)
        hybrid_ranks[hybrid_order] = np.arange(1, len(pool) + 1)

        candidates = [factors.by_id.get(int(factors.ids[pos]), ("", "")) for pos in pool]

        # Sort the union of both top lists by name so the judge can't tell which model ranked it.
        chosen = set(raw_order[: args.top]) | set(hybrid_order[: args.top])
        for j in sorted(chosen, key=lambda k: (candidates[k][1] or "").casefold()):
            mbid, name = candidates[j]
            rows.append(
                {
                    "seed_mbid": seed_mbid,
                    "seed_artist": seed_name,
                    "candidate_mbid": mbid,
                    "candidate_artist": name or mbid,
                    "relevance": "",
                }
            )
            details.append(
                {
                    "seed_mbid": seed_mbid,
                    "candidate_mbid": mbid,
                    "als_score": f"{raw[j]:.7f}",
                    "genre_score": f"{genre_sim[j]:.7f}" if np.isfinite(genre_sim[j]) else "",
                    "confidence": f"{conf[j]:.7f}",
                    "raw_rank": int(raw_ranks[j]),
                    "hybrid_rank": int(hybrid_ranks[j]),
                    "hybrid_score": f"{hybrid[j]:.7f}",
                }
            )
        print(f"{seed_name}: {len(chosen)} unique candidates from top {args.top} of each model")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(args.details, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_mbid", "candidate_mbid", *DETAILS])
        writer.writeheader()
        writer.writerows(details)
    print(f"Judge: {args.output}\nKeep for scoring: {args.details}")


def score(args):
    with open(args.judgments, newline="", encoding="utf-8") as f:
        judgments = list(csv.DictReader(f))
    with open(args.details, newline="", encoding="utf-8") as f:
        detail_rows = list(csv.DictReader(f))
    if not judgments or not detail_rows:
        sys.exit("Empty judgments or details file")

    labels = {}
    for r in judgments:
        key = (r["seed_mbid"], r["candidate_mbid"])
        if key in labels:
            sys.exit(f"Duplicate judgment: {key}")
        try:
            val = int(r["relevance"])
            if str(val) != r["relevance"].strip() or val not in (0, 1, 2, 3):
                raise ValueError
        except (ValueError, TypeError):
            sys.exit(
                f"Missing/invalid relevance (0-3) for {r['seed_artist']} -> {r['candidate_artist']}"
            )
        labels[key] = val

    groups = {}
    for r in detail_rows:
        key = (r["seed_mbid"], r["candidate_mbid"])
        if key not in labels:
            sys.exit(f"Missing judgment for {key}")
        groups.setdefault(r["seed_mbid"], []).append((r, labels[key]))

    raw_scores, hybrid_scores = [], []
    for seed, items in groups.items():
        # Judged candidates are the union of both top-k lists, so the ideal DCG
        # comes from the union while each model is scored on its own top k.
        ideal_grades = sorted((g for _, g in items), reverse=True)
        ideal = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades[: args.k]))
        results = []
        for rank_key in ("raw_rank", "hybrid_rank"):
            ordered = sorted(items, key=lambda pair, key=rank_key: int(pair[0][key]))
            top = [(int(r[rank_key]), g) for r, g in ordered if int(r[rank_key]) <= args.k]
            if len(top) != args.k:
                sys.exit(
                    f"{seed}: need all top {args.k} judgments for {rank_key}; found {len(top)}"
                )
            dcg = sum((2**g - 1) / math.log2(rank + 1) for rank, g in top)
            results.append(dcg / ideal if ideal else 0.0)
        raw_scores.append(results[0])
        hybrid_scores.append(results[1])
        print(
            f"{seed}: raw={results[0]:.4f} hybrid={results[1]:.4f} "
            f"delta={results[1] - results[0]:+.4f}"
        )
    print(
        f"Macro NDCG@{args.k}: raw={np.mean(raw_scores):.4f} hybrid={np.mean(hybrid_scores):.4f} "
        f"delta={np.mean(hybrid_scores) - np.mean(raw_scores):+.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="Export blinded judgments and separate score details")
    e.add_argument("artists", nargs="+", help="Seed artist names")
    e.add_argument("--top", type=int, default=40)
    e.add_argument("--pool", type=int, default=500)
    e.add_argument("--weight", type=float, default=0.2)
    e.add_argument("--neutral", type=float, default=0.3)
    e.add_argument("--prior-votes", type=float, default=10)
    e.add_argument("--output", type=Path, default=DEFAULT_JUDGMENTS)
    e.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    s = sub.add_parser("score", help="Calculate NDCG from completed judgments")
    s.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    s.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    s.add_argument("-k", type=int, default=20)
    args = parser.parse_args()

    if args.command == "export":
        if (
            args.top < 1
            or args.pool < args.top
            or args.prior_votes <= 0
            or not 0 <= args.weight <= 1
        ):
            parser.error("Require 1 <= top <= pool, positive prior-votes, and weight in [0,1]")
        if args.output.resolve() == args.details.resolve():
            parser.error("Output and details paths must differ")
        export(args)
    else:
        if args.k < 1:
            parser.error("k must be positive")
        score(args)


if __name__ == "__main__":
    main()
