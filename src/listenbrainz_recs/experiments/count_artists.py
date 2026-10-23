"""Summarize how many artists each user has in the raw ListenBrainz statistics dump."""

import json
from collections import Counter

from listenbrainz_recs.paths import ARTISTS_ALL_TIME


def main():
    lengths = Counter()
    total_interactions = 0
    max_artists = 0
    with open(ARTISTS_ALL_TIME) as f:
        for line in f:
            n = len(json.loads(line)["data"])
            lengths[n] += 1
            total_interactions += n
            max_artists = max(max_artists, n)

    users = sum(lengths.values())
    print(f"Users: {users:,}")
    print(f"Interactions: {total_interactions:,}")
    print(f"Average artists/user: {total_interactions / users:.1f}")
    print(f"Maximum artists/user: {max_artists:,}")
    print("Most common artist counts:", lengths.most_common(15))


if __name__ == "__main__":
    main()
