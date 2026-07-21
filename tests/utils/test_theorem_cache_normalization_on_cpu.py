"""CPU-only tests for theorem-cache-key alpha-normalization.

The cache shares one proof outcome across every theorem whose normalized text is
equal, so these tests guard the zero-false-positive mandate: theorems that differ
in any provability-relevant way (numerals, operators, term structure, variable
identity, types, tactics) MUST keep distinct keys, while generated theorems that
differ only in their declared free-variable names or whitespace share one key.
"""

from verl.utils.isabelle_utils._server_pool.theorem_cache import (
    normalize_theorem_text as N,
)


# ---- theorems that SHOULD share one key (safe reuse) ----

def test_free_variable_renaming_shares_one_key():
    a = ('theorem chk:\n  fixes pv_x :: int\n'
         '  assumes g0: "pv_x = 5"\n  shows "pv_x + 3 = 8"')
    b = ('theorem chk:\n  fixes pv_answer :: int\n'
         '  assumes g0: "pv_answer = 5"\n  shows "pv_answer + 3 = 8"')
    assert N(a) == N(b)


def test_whitespace_collapse_shares_one_key():
    one_line = 'theorem chk: fixes pv_x :: int assumes g0: "pv_x = 5" shows "pv_x + 3 = 8"'
    indented = ('theorem chk:\n'
                '  fixes pv_x :: int\n'
                '  assumes g0: "pv_x = 5"\n'
                '  shows "pv_x + 3 = 8"')
    assert N(one_line) == N(indented)


def test_consistent_multi_variable_renaming_shares_key():
    a = ('theorem chk:\n  fixes pv_a pv_b :: int\n'
         '  assumes g0: "pv_a = 2" and g1: "pv_b = 3"\n'
         '  shows "pv_a * pv_b = 6"')
    b = ('theorem chk:\n  fixes pv_p pv_q :: int\n'
         '  assumes g0: "pv_p = 2" and g1: "pv_q = 3"\n'
         '  shows "pv_p * pv_q = 6"')
    assert N(a) == N(b)


def test_multiple_fixes_clauses_share_key_after_renaming():
    a = ('theorem chk:\n  fixes pv_a :: int\n  fixes pv_b :: real\n'
         '  assumes g0: "pv_a = 2" and g1: "pv_b = 3"\n'
         '  shows "real pv_a + pv_b = 5"')
    b = ('theorem chk:\n  fixes pv_x :: int\n  fixes pv_y :: real\n'
         '  assumes g0: "pv_x = 2" and g1: "pv_y = 3"\n'
         '  shows "real pv_x + pv_y = 5"')
    assert N(a) == N(b)


# ---- theorems that MUST keep distinct keys (else false positive) ----

def test_different_numerals_keep_distinct_keys():
    proved = 'fixes pv_x :: int assumes "pv_x = 5" shows "pv_x + 3 = 8"'
    wrong = 'fixes pv_x :: int assumes "pv_x = 5" shows "pv_x + 3 = 9"'
    assert N(proved) != N(wrong)


def test_different_operator_keeps_distinct_keys():
    plus = 'fixes pv_x :: int assumes "pv_x = 5" shows "pv_x + 3 = 8"'
    minus = 'fixes pv_x :: int assumes "pv_x = 5" shows "pv_x - 3 = 8"'
    assert N(plus) != N(minus)


def test_variable_identity_is_preserved_not_just_names():
    one_var = N('fixes pv_x :: int shows "pv_x + pv_x = 10"')
    two_vars = N('fixes pv_x pv_y :: int shows "pv_x + pv_y = 10"')
    assert one_var != two_vars


def test_prefix_colliding_declared_names_are_distinct_variables():
    two_vars = N('fixes pv_x1 pv_x11 :: int shows "pv_x1 + pv_x11 = 10"')
    renamed_two = N('fixes pv_a pv_b :: int shows "pv_a + pv_b = 10"')
    one_var = N('fixes pv_x1 :: int shows "pv_x1 + pv_x1 = 10"')
    assert two_vars == renamed_two
    assert two_vars != one_var


def test_non_pv_tokens_are_untouched():
    assert N('fixes pv_x :: int shows "pv_x = 5"') != N('fixes pv_x :: int shows "pv_x = 6"')
    assert N('fixes pv_x :: int shows "pv_x = 5" by linarith') != N('fixes pv_x :: int shows "pv_x = 5" by presburger')
    assert N('fixes pv_x :: int') != N('fixes pv_x :: real')


def test_undeclared_pv_prefixed_token_is_not_renamed():
    # A direct theorem may mention an Isabelle constant or locale entity whose
    # spelling happens to start with pv_; only theorem-header fixes are alpha-renamed.
    a = 'theorem chk: shows "pv_external = 5" by simp'
    b = 'theorem chk: shows "pv_other = 5" by simp'
    assert N(a) != N(b)


def test_declared_and_undeclared_pv_tokens_cannot_collapse():
    declared = 'theorem chk: fixes pv_external :: int shows "pv_external = 5"'
    undeclared = 'theorem chk: shows "pv_external = 5"'
    assert N(declared) != N(undeclared)


def test_token_spacing_is_not_altered():
    assert N('fixes pv_x :: int') != N('fixes pv_x::int')


# ---- function-level properties ----

def test_normalization_is_deterministic_and_does_not_mutate_input():
    original = 'fixes pv_x :: int assumes "pv_x = 5" shows "pv_x + 3 = 8"'
    once = N(original)
    assert N(original) == once
    assert original == 'fixes pv_x :: int assumes "pv_x = 5" shows "pv_x + 3 = 8"'


def test_placeholders_follow_declaration_order():
    a = ('fixes pv_first pv_second :: int '
         'assumes "pv_second = 2" "pv_first = 1" '
         'shows "pv_first < pv_second"')
    b = ('fixes pv_alpha pv_beta :: int '
         'assumes "pv_beta = 2" "pv_alpha = 1" '
         'shows "pv_alpha < pv_beta"')
    assert N(a) == N(b)
