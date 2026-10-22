"""Ten link-negative sets and score aggregation for the user-run HD notebook."""
import random

import numpy as np

from hd_prediction import DISEASE

NEGATIVE_SEEDS = tuple(range(10))


def sample_negative_sets(positives, seed0_negatives, node_types):
    """Preserve seed 0; sample later balanced sets without cross-set overlap."""
    known = set(positives)
    if not known or len(known) != len(positives):
        raise ValueError("Positive pairs must be nonempty and unique")
    for drug, disease in positives + seed0_negatives:
        if node_types.get(drug) != "drug" or node_types.get(disease) != "disease":
            raise ValueError("Unexpected drug/disease endpoint")
    if (len(seed0_negatives) != len(positives)
            or len(set(seed0_negatives)) != len(seed0_negatives)
            or known.intersection(seed0_negatives)):
        raise ValueError("Invalid saved seed-0 negatives")
    drugs = sorted(node for node, kind in node_types.items() if kind == "drug")
    diseases = sorted(node for node, kind in node_types.items() if kind == "disease")
    candidates = [(drug, disease) for drug in drugs for disease in diseases
                  if (drug, disease) not in known]
    if len(candidates) < len(positives) * len(NEGATIVE_SEEDS):
        raise ValueError("Insufficient unknown pairs for ten disjoint negative sets")
    if random.Random(0).sample(candidates, len(positives)) != seed0_negatives:
        raise ValueError("Saved seed-0 negatives differ from the canonical seed-0 sample")
    negative_sets = {0: list(seed0_negatives)}
    selected = set(seed0_negatives)
    for seed in NEGATIVE_SEEDS[1:]:
        candidates = [pair for pair in candidates if pair not in selected]
        negative_sets[seed] = random.Random(seed).sample(candidates, len(positives))
        selected.update(negative_sets[seed])
    return negative_sets


def aggregate_scores(drugs, scores_by_seed, names, positives, negative_sets, disease=DISEASE):
    """Rank mean scores across all ten classifiers; retain training exposure."""
    if set(scores_by_seed) != set(NEGATIVE_SEEDS) or set(negative_sets) != set(NEGATIVE_SEEDS):
        raise ValueError("All ten negative-set seeds are required before aggregation")
    if not drugs or len(drugs) != len(set(drugs)):
        raise ValueError("Drug IDs must be nonempty and unique")
    scores = np.asarray([scores_by_seed[seed] for seed in NEGATIVE_SEEDS], dtype=float)
    if (scores.shape != (10, len(drugs)) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Invalid or misaligned prediction scores")
    known = {drug for drug, indication in positives if indication == disease}
    if not known <= set(drugs):
        raise ValueError("A known target drug is missing from candidates")
    pools = [set(negative_sets[seed]) for seed in NEGATIVE_SEEDS]
    means, deviations = scores.mean(axis=0), scores.std(axis=0, ddof=1)
    ranked = []
    for index, drug in enumerate(drugs):
        exposure = sum((drug, disease) in pool for pool in pools)
        ranked.append(dict(
            drug=drug, drug_name=names.get(drug) or drug, disease=disease,
            disease_name="Huntington disease", mean_score=float(means[index]),
            sd_score=float(deviations[index]), n_negative_sets=len(NEGATIVE_SEEDS),
            known_in_msi=drug in known, training_negative_count=exposure,
            training_negative_fraction=exposure / len(NEGATIVE_SEEDS),
        ))
    ranked.sort(key=lambda row: (-row["mean_score"], row["drug"]))
    candidates, known_rows = [], []
    for rank, row in enumerate(ranked, 1):
        row["rank_all"] = rank
        row["candidate_rank"] = "" if row["known_in_msi"] else len(candidates) + 1
        (known_rows if row["known_in_msi"] else candidates).append(dict(row))
    return ranked, candidates, known_rows
