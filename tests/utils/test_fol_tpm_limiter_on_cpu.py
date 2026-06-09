from pathlib import Path

from verl.utils.fol_utils import common


def test_openai_tpm_budget_can_be_reduced_after_request(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "fol_tpm_state.json"
    monkeypatch.setenv("FOL_OPENAI_TPM_STATE_PATH", str(state_path))

    reservation = common._reserve_openai_tpm_budget(100, 90)
    assert reservation is not None

    common._update_openai_tpm_budget(reservation, 10)

    second = common._reserve_openai_tpm_budget(100, 95)
    assert second is not None

    common._update_openai_tpm_budget(second, release=True)

