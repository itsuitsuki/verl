"""CPU-only tests for Isabelle training metric aggregation."""

from verl.trainer.ppo.isabelle_metrics import (
    ISABELLE_PROFILE_METRIC_MAP,
    aggregate_mapped_metrics,
)


def test_profile_keys_and_cumulative_cleanup_are_aggregated():
    batch = {
        "isabelle_judge_http_wall_time": [1.0, 3.0],
        "isabelle_translate_validate_wall_time": [2.0, 6.0],
        "isabelle_prove_queue_time": [4.0, 8.0],
        "isabelle_prove_run_time": [5.0, 9.0],
        "isabelle_reward_wall_time": [10.0, 14.0],
        "isabelle_external_solver_reaps": [2, 7],
    }
    metrics = aggregate_mapped_metrics(
        batch,
        ISABELLE_PROFILE_METRIC_MAP,
        {"isabelle_external_solver_reaps"},
    )
    assert metrics["isabelle/judge_http_wall_s/mean"] == 2.0
    assert metrics["isabelle/translate_validate_wall_s/max"] == 6.0
    assert metrics["isabelle/prove_queue_s/min"] == 4.0
    assert metrics["isabelle/prove_run_s/mean"] == 7.0
    assert metrics["isabelle/reward_wall_s/max"] == 14.0
    assert metrics["isabelle/external_solver_reaps/current"] == 7.0
