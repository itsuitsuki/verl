theory Math_Verify_Base
  imports
    "HOL-Analysis.Analysis"
    "HOL-Real_Asymp.Real_Asymp"
    "HOL-Library.Sum_of_Squares"
    "HOL-Library.Code_Target_Numeral"
    "HOL-Number_Theory.Number_Theory"
    "HOL-Decision_Procs.Approximation"
begin

(* Quadrant-trig meta-theorems (2026-07-14). "angle a in a given quadrant,
   tan a = t, find sin a / cos a" is solved by these value-independent facts:
   establish the sign of cos a from the quadrant, instantiate, then let simp
   evaluate the (possibly irrational, e.g. 2/sqrt 5) constant. div_sqrt_eq
   rationalises A/sqrt B = C to the pure check A^2 = C^2*B, so rational and
   irrational answers close identically. Referenced by the step-reward engine's
   trig recipe (verl/utils/isabelle_utils/tactics.py). *)

lemma div_sqrt_eq:
  fixes A B C :: real
  assumes b: "0 < B" and a: "0 \<le> A" and c: "0 \<le> C" and sq: "A^2 = C^2 * B"
  shows "A / sqrt B = C"
proof -
  have sb: "sqrt B > 0" using b by simp
  have "(A / sqrt B)^2 = A^2 / (sqrt B)^2" by (simp add: power_divide)
  also have "(sqrt B)^2 = B" using b by simp
  finally have e2: "(A / sqrt B)^2 = A^2 / B" by simp
  have e3: "A^2 / B = C^2" using sq b by (simp add: field_simps)
  have nn: "A / sqrt B \<ge> 0" using a sb by simp
  from e2 e3 have "(A / sqrt B)^2 = C^2" by simp
  thus ?thesis using nn c by (metis power2_eq_imp_eq)
qed

lemma cos_abs_from_tan:
  fixes a t :: real
  assumes cnz: "cos a \<noteq> 0" and tan_eq: "tan a = t"
  shows "abs (cos a) = 1 / sqrt (t^2 + 1)"
proof -
  have pos: "t^2 + 1 > 0" using zero_le_power2[of t] by linarith
  have t_def: "sin a = t * cos a" using tan_eq cnz unfolding tan_def by (simp add: field_simps)
  have "(t * cos a)^2 + (cos a)^2 = 1" using sin_cos_squared_add[of a] t_def by simp
  hence "t^2 * (cos a)^2 + (cos a)^2 = 1" by (simp add: power_mult_distrib)
  hence "(t^2 + 1) * (cos a)^2 = 1" by (simp add: algebra_simps)
  hence cos2: "(cos a)^2 = 1 / (t^2 + 1)" using pos by (simp add: field_simps)
  have "sqrt ((cos a)^2) = sqrt (1 / (t^2 + 1))" using cos2 by simp
  thus ?thesis by (simp add: real_sqrt_abs real_sqrt_divide)
qed

lemma cos_from_tan_cneg:
  fixes a t :: real assumes "cos a < 0" and "tan a = t"
  shows "cos a = - 1 / sqrt (t^2 + 1)"
  using cos_abs_from_tan[of a t] assms by simp

lemma cos_from_tan_cpos:
  fixes a t :: real assumes "cos a > 0" and "tan a = t"
  shows "cos a = 1 / sqrt (t^2 + 1)"
  using cos_abs_from_tan[of a t] assms by simp

lemma sin_from_tan_cneg:
  fixes a t :: real assumes c: "cos a < 0" and "tan a = t"
  shows "sin a = - t / sqrt (t^2 + 1)"
proof -
  have "sin a = t * cos a" using assms(2) c unfolding tan_def by (simp add: field_simps)
  thus ?thesis using cos_from_tan_cneg[OF c assms(2)] by simp
qed

lemma sin_from_tan_cpos:
  fixes a t :: real assumes c: "cos a > 0" and "tan a = t"
  shows "sin a = t / sqrt (t^2 + 1)"
proof -
  have "sin a = t * cos a" using assms(2) c unfolding tan_def by (simp add: field_simps)
  thus ?thesis using cos_from_tan_cpos[OF c assms(2)] by simp
qed

lemma sin_pos_upper: "0 < a \<Longrightarrow> a < pi \<Longrightarrow> sin (a::real) > 0"
  by (intro sin_gt_zero) auto

lemma sin_neg_lower: "pi < a \<Longrightarrow> a < 2*pi \<Longrightarrow> sin (a::real) < 0"
proof -
  assume h: "pi < a" "a < 2*pi"
  have "sin a = - sin (a - pi)" by (simp add: sin_diff)
  moreover have "sin (a - pi) > 0" using h by (intro sin_gt_zero) auto
  ultimately show ?thesis by simp
qed

lemma cos_pos_q1: "0 < a \<Longrightarrow> a < pi/2 \<Longrightarrow> cos (a::real) > 0"
  by (intro cos_gt_zero_pi) auto

lemma cos_neg_q2: "pi/2 < a \<Longrightarrow> a < pi \<Longrightarrow> cos (a::real) < 0"
proof -
  assume h: "pi/2 < a" "a < pi"
  hence "cos a < cos (pi/2)" by (intro cos_monotone_0_pi) auto
  thus ?thesis by (simp add: cos_pi_half)
qed

lemma cos_neg_q3: "pi < a \<Longrightarrow> a < 3*pi/2 \<Longrightarrow> cos (a::real) < 0"
proof -
  assume h: "pi < a" "a < 3*pi/2"
  have "cos a = - cos (a - pi)" by (simp add: cos_diff)
  moreover have "cos (a - pi) > 0" using h by (intro cos_gt_zero_pi) auto
  ultimately show ?thesis by simp
qed

lemma cos_pos_q4: "3*pi/2 < a \<Longrightarrow> a < 2*pi \<Longrightarrow> cos (a::real) > 0"
proof -
  assume h: "3*pi/2 < a" "a < 2*pi"
  have "cos a = - cos (a - pi)" by (simp add: cos_diff)
  moreover have "cos (a - pi) < 0" using h cos_neg_q2[of "a - pi"] by auto
  ultimately show ?thesis by simp
qed


(* ==== Pigeonhole meta-theorems (2026-07-14) ============================
   min_rep n k = fewest draws over n categories that FORCE some category to
   hold >= k = n*(k-1)+1. CERTIFIED minimal by min_rep_correct (sufficiency +
   minimality, k>=2). The step-reward engine's guarantee_pair(n)=min_rep n 2
   and guarantee_same(n,k)=min_rep n k (verl/utils/isabelle_utils/pyexpr.py)
   let the judge supply only the category count; the pigeonhole "+1" content is
   fixed and sound. min_rep is executable and [simp]-unfolds to the closed form. *)

definition guarantees_rep :: "nat \<Rightarrow> nat \<Rightarrow> nat \<Rightarrow> bool" where
  "guarantees_rep m n k \<longleftrightarrow>
     (\<forall>f. (\<forall>i<m. f i < n) \<longrightarrow> (\<exists>c<n. k \<le> card {i. i < m \<and> f i = c}))"

lemma guarantees_rep_suff:
  assumes "n * (k - 1) < m"
  shows "guarantees_rep m n k"
  unfolding guarantees_rep_def
proof (intro allI impI)
  fix f :: "nat \<Rightarrow> nat"
  assume A: "\<forall>i<m. f i < n"
  show "\<exists>c<n. k \<le> card {i. i < m \<and> f i = c}"
  proof (rule ccontr)
    assume neg: "\<not> (\<exists>c<n. k \<le> card {i. i < m \<and> f i = c})"
    let ?A = "\<lambda>c. {i. i < m \<and> f i = c}"
    have part: "{..<m} = (\<Union>c<n. ?A c)" using A by auto
    have "m = card {..<m}" by simp
    also have "\<dots> = (\<Sum>c<n. card (?A c))"
      unfolding part by (subst card_UN_disjoint) auto
    also have "\<dots> \<le> (\<Sum>c<n. k - 1)"
    proof (rule sum_mono)
      fix c assume "c \<in> {..<n}"
      then have "\<not> k \<le> card (?A c)" using neg by auto
      then show "card (?A c) \<le> k - 1" by simp
    qed
    also have "\<dots> = n * (k - 1)" by simp
    finally have "m \<le> n * (k - 1)" .
    with assms show False by simp
  qed
qed

lemma guarantees_rep_necess:
  assumes "m \<le> n * (k - 1)" and "2 \<le> k"
  shows "\<not> guarantees_rep m n k"
proof -
  define d where "d = k - 1"
  have d1: "1 \<le> d" using assms(2) unfolding d_def by simp
  have fib: "card {i. i < m \<and> i div d = c} < k" for c
  proof -
    have inj: "inj_on (\<lambda>i. i mod d) {i. i < m \<and> i div d = c}"
    proof (rule inj_onI)
      fix x y
      assume "x \<in> {i. i < m \<and> i div d = c}" and "y \<in> {i. i < m \<and> i div d = c}"
        and eqmod: "x mod d = y mod d"
      then have "x div d = y div d" by simp
      with eqmod show "x = y" by (metis div_mult_mod_eq)
    qed
    have sub: "(\<lambda>i. i mod d) ` {i. i < m \<and> i div d = c} \<subseteq> {..<d}"
      using d1 by auto
    have "card {i. i < m \<and> i div d = c} \<le> card {..<d}"
      by (rule card_inj_on_le[OF inj sub finite_lessThan])
    also have "\<dots> = d" by simp
    also have "d < k" using assms(2) unfolding d_def by simp
    finally show ?thesis .
  qed
  have bnd: "\<forall>i<m. i div d < n"
  proof (intro allI impI)
    fix i assume "i < m"
    then have "i < n * d" using assms(1) unfolding d_def by (meson less_le_trans)
    then show "i div d < n" by (rule less_mult_imp_div_less)
  qed
  have noc: "\<not> (\<exists>c<n. k \<le> card {i. i < m \<and> i div d = c})"
    using fib by (meson leD)
  show ?thesis
    unfolding guarantees_rep_def not_all
  proof (rule exI[of _ "\<lambda>i. i div d"])
    show "\<not> ((\<forall>i<m. i div d < n) \<longrightarrow> (\<exists>c<n. k \<le> card {i. i < m \<and> i div d = c}))"
      using bnd noc by blast
  qed
qed

definition min_rep :: "nat \<Rightarrow> nat \<Rightarrow> nat" where
  "min_rep n k = n * (k - 1) + 1"

lemma min_rep_correct:
  assumes "2 \<le> k"
  shows "guarantees_rep (min_rep n k) n k
         \<and> (\<forall>m. guarantees_rep m n k \<longrightarrow> min_rep n k \<le> m)"
proof
  show "guarantees_rep (min_rep n k) n k"
    unfolding min_rep_def by (rule guarantees_rep_suff) simp
next
  show "\<forall>m. guarantees_rep m n k \<longrightarrow> min_rep n k \<le> m"
  proof (intro allI impI)
    fix m assume "guarantees_rep m n k"
    with guarantees_rep_necess assms have "\<not> m \<le> n * (k - 1)" by blast
    then show "min_rep n k \<le> m" unfolding min_rep_def by simp
  qed
qed

declare min_rep_def [simp]


end
