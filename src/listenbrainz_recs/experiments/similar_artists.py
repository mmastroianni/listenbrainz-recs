"""Print an artist's nearest neighbors by ALS cosine similarity."""

import argparse
import sys

from listenbrainz_recs.artists import load_artist_factors, resolve_artist, top_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artist",
        nargs="?",
        default="Braxton",
        help="Artist name, partial name, nickname, or MusicBrainz ID",
    )
    parser.add_argument("-n", "--number", type=int, default=20)
    args = parser.parse_args()
    if args.number < 1:
        parser.error("--number must be positive")

    factors = load_artist_factors("SimilarArtists")
    target_id = resolve_artist(args.artist, factors.artists)
    if target_id is None:
        sys.exit(1)
    target_pos = factors.position(target_id)
    if target_pos is None:
        sys.exit("Artist has no trained factor vector")

    similarities = factors.similarities(target_pos)
    top = top_positions(similarities, min(args.number, len(similarities) - 1))

    print(f"\nArtists similar to {factors.by_id[target_id][1]}:\n")
    for rank, pos in enumerate(top, 1):
        print(f"{rank:2d}. {factors.label(pos):40s} {similarities[pos]:.4f}")


if __name__ == "__main__":
    main()
