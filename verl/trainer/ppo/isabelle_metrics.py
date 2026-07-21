"""Isabelle reward metric names and trainer aggregation."""

import numpy as np


ISABELLE_PROFILE_METRIC_MAP = {
    "isabelle_judge_http_wall_time": "isabelle/judge_http_wall_s",
    "isabelle_translate_validate_wall_time": "isabelle/translate_validate_wall_s",
    "isabelle_prove_queue_time": "isabelle/prove_queue_s",
    "isabelle_prove_run_time": "isabelle/prove_run_s",
    "isabelle_reward_wall_time": "isabelle/reward_wall_s",
    "isabelle_external_solver_reaps": "isabelle/external_solver_reaps",
}


def aggregate_mapped_metrics(non_tensor_batch, metric_map, cumulative_gauges=()):
    """Aggregate mapped arrays as mean, max, min, and selected current values."""
    metrics = {}
    for batch_key, metric_prefix in metric_map.items():
        if batch_key not in non_tensor_batch:
            continue
        values = np.asarray(non_tensor_batch[batch_key], dtype=np.float32)
        metrics[f"{metric_prefix}/mean"] = float(np.mean(values))
        metrics[f"{metric_prefix}/max"] = float(np.max(values))
        metrics[f"{metric_prefix}/min"] = float(np.min(values))
        if batch_key in cumulative_gauges:
            metrics[f"{metric_prefix}/current"] = float(np.max(values))
    return metrics
