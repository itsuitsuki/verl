# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        # score (training reward) = math-verify boxed-gated: uniform reward
        # basis across all three training sources (gsm8k/MATH/Big-Math) and
        # enforces the \boxed{} contract (bare "#### 18" earns nothing --
        # empirically a non-issue: feas step200 acc_mathverify 87.2 > regex
        # acc 86.8, the model boxes everything). acc stays the historical
        # regex scorer so val-core remains comparable with prior runs.
        from . import gsm8k, math_verify

        _g = gsm8k.compute_score(solution_str, ground_truth)
        _mv = math_verify.compute_score_boxed(solution_str, ground_truth)
        res = {"score": _mv, "acc": _g, "acc_mathverify": _mv}
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        # score (training reward) = math-verify boxed-gated: is_equiv's
        # string normalization mis-scores ~5% of correct answers on MATH
        # (feas step200: acc 66.0 vs acc_mathverify 70.6) and that false
        # negative is pure reward noise. acc stays is_equiv (Hendrycks /
        # lm-eval-harness standard) so val-core remains comparable with the
        # community and our own history; both graders logged on every val.
        from . import math_reward, math_verify

        _equiv = math_reward.compute_score(solution_str, ground_truth)
        _mv = math_verify.compute_score_boxed(solution_str, ground_truth)
        res = {"score": _mv, "acc": _equiv, "acc_mathverify": _mv}
    elif data_source in [
        "open-r1/Big-Math-RL-Verified-Processed",
        "open-r1/DAPO-Math-17k-Processed",
        "zwhe99/DeepMath-103K",
        "math-ai/aime24",
        "math-ai/aime25",
        "math-ai/amc23",
        "math-ai/minervamath",
        "math-ai/olympiadbench",
    ]:
        # Big-Math (train) + the non-MATH eval benches: ground truths mix
        # currency ($2.50), symbolic (2\sqrt{2}) and fraction forms;
        # math-verify's sympy equivalence handles all of them (verified
        # 2026-07-05: 7/7 positive, 6/6 negative smoke cases), where
        # math_reward.is_equiv's string normalization would misjudge.
        # Boxed-gated to keep the \boxed{} format contract fail-closed.
        # Same key set as gsm8k/MATH routes (batch collation uniformity).
        from . import math_verify

        _mv = math_verify.compute_score_boxed(solution_str, ground_truth)
        res = {"score": _mv, "acc": _mv, "acc_mathverify": _mv}
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source in [
        "logiqa",
        "lucasmccabe/logiqa",
        "reclor",
        "voidful/ReClor",
        "ar_lsat",
        "olegbask/AR-LSAT",
    ]:
        from . import logiqa

        res = logiqa.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


def default_compute_score_image(
    data_source,
    solution_image,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_image (Image.Image or torch.Tensor): The solution image to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "jpeg_compressibility":
        from . import jpeg_compressibility

        res = jpeg_compressibility.compute_score(solution_image)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    if reward_name == "visual":
        return default_compute_score_image
    else:
        return default_compute_score


__all__ = ["default_compute_score"]
