"""MusicBrainz genre vectors and genre-aware reranking of ALS neighbors."""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

from listenbrainz_recs.paths import ARTIST_TAGS


class Genres(NamedTuple):
    vectors: csr_matrix  # L2-normalized TF-IDF genre vectors, one row per tagged artist
    rows: dict[str, int]  # artist MBID -> row in vectors
    votes: np.ndarray  # total positive genre votes per row


def load_genres() -> Genres:
    tags = pd.read_parquet(ARTIST_TAGS, columns=["artist_mbid", "tag", "tag_count", "is_genre"])
    tags = tags.loc[tags.is_genre & (tags.tag_count > 0)].copy()
    tags = tags.groupby(["artist_mbid", "tag"], as_index=False)["tag_count"].sum()
    artist_codes, mbids = pd.factorize(tags.artist_mbid)
    genre_codes, genres = pd.factorize(tags.tag)
    counts = csr_matrix(
        (np.log1p(tags.tag_count.to_numpy(dtype=np.float32)), (artist_codes, genre_codes)),
        shape=(len(mbids), len(genres)),
    )
    vectors = TfidfTransformer(norm="l2").fit_transform(counts)
    # Use raw positive votes as a rough measure of tag reliability.
    votes = np.bincount(
        artist_codes,
        weights=tags.tag_count.to_numpy(dtype=np.float64),
        minlength=len(mbids),
    )
    return Genres(vectors, {mbid: i for i, mbid in enumerate(mbids)}, votes)


def genre_similarity(
    genres: Genres,
    seed_mbid: str,
    candidate_mbids: list[str | None],
    prior_votes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Genre cosine similarity and tag confidence for each candidate.

    Similarity is NaN and confidence 0 wherever the seed or candidate is untagged.
    Confidence is the geometric mean of votes / (votes + prior_votes) for both artists.
    """
    similarity = np.full(len(candidate_mbids), np.nan)
    confidence = np.zeros(len(candidate_mbids))
    seed_row = genres.rows.get(seed_mbid)
    if seed_row is None:
        return similarity, confidence
    seed_conf = genres.votes[seed_row] / (genres.votes[seed_row] + prior_votes)
    for i, mbid in enumerate(candidate_mbids):
        row = genres.rows.get(mbid)
        if row is not None:
            similarity[i] = float(genres.vectors[seed_row].multiply(genres.vectors[row]).sum())
            candidate_conf = genres.votes[row] / (genres.votes[row] + prior_votes)
            confidence[i] = np.sqrt(seed_conf * candidate_conf)
    return similarity, confidence


def confidence_adjusted(
    als: np.ndarray,
    similarity: np.ndarray,
    confidence: np.ndarray,
    weight: float,
    neutral: float,
) -> np.ndarray:
    """Add a confidence-scaled genre adjustment around a neutral baseline.

    Untagged artists have zero confidence, so their ALS score is unchanged.
    """
    return als + weight * confidence * (np.nan_to_num(similarity, nan=neutral) - neutral)


def linear_blend(als: np.ndarray, similarity: np.ndarray, weight: float) -> np.ndarray:
    """Interpolate ALS and genre scores; keep raw ALS where genre data is missing."""
    blended = als.copy()
    known = np.isfinite(similarity)
    blended[known] = (1 - weight) * als[known] + weight * similarity[known]
    return blended


def neighborhood_coherence(genres: Genres, rows: list[int]) -> dict[str, float] | None:
    """Cosine dispersion and mean pairwise similarity of tagged neighbors."""
    n = len(rows)
    if n < 2:
        return None
    centroid = np.asarray(genres.vectors[rows].mean(axis=0)).ravel()
    norm_sq = float(centroid @ centroid)
    return {
        "dispersion": 1.0 - np.sqrt(norm_sq),
        "pairwise_similarity": (n * norm_sq - 1.0) / (n - 1),
    }
