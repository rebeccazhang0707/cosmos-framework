"""Small, fixed, outcome-independent RoboDojo SR protocol:15 episodes/family."""

import math

SEEDS = (0, 1, 2)
EPISODES_PER_SEED = 5
EPISODES_PER_FAMILY = len(SEEDS) * EPISODES_PER_SEED
TOTAL_EPISODES = 4 * EPISODES_PER_FAMILY


def evaluation_cells():
    cells = []
    for seed in SEEDS:
        standard = 3 if seed % 2 == 0 else 2
        randomized = EPISODES_PER_SEED - standard
        cells.extend(
            [
                ("classify_objects", seed, 5),
                ("insert_tubes", seed, 5),
                ("stack_bowls", seed, standard),
                ("stack_bowls_random", seed, randomized),
                ("push_T", seed, standard),
                ("push_T_random", seed, randomized),
            ]
        )
    return cells


def native_prefix_metrics(native, quota):
    """Reuse the first quota valid episodes without rewriting native history.

    Older runs may already exceed the new quota. Their later outcomes must not
    enter the smaller protocol's SR or process score. Selection never uses the
    success flag; raw native totals remain available for auditing.
    """
    count = int(native.get("eval_time", 0))
    details = native.get("details", {})
    if count < 0 or count != len(details) or quota < 1:
        raise ValueError("Incomplete native episode accounting or invalid quota")
    records = sorted(details.items(), key=lambda item: int(item[0]))[:quota]
    scores = [float(row["score"]) for _, row in records]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Nonfinite native episode score")
    return dict(
        episodes=len(records),
        native_episodes=count,
        successes=sum(bool(row["success"]) for _, row in records),
        score=100 * sum(scores) / len(scores) if scores else None,
        selected_episode_indices=[int(key) for key, _ in records],
        selected_layout_ids=[row.get("layout_id") for _, row in records],
    )
