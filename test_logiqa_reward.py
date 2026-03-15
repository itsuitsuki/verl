from verl.utils.reward_score import default_compute_score

def test_logiqa_reward():
    test_cases = [
        ("The answer is \\boxed{A}.", "A", 1.0),
        ("The correct answer is B", "B", 1.0),
        ("Selected Option(C)", "C", 1.0),
        ("Answer: D", "D", 1.0),
        ("Option A is the correct answer", "A", 1.0),
        ("Thinking process... The answer is Option B.", "B", 1.0),
        ("The answer is \\boxed{A}.", "B", 0.0),
        ("No answer given here.", "A", 0.0),
        ("Multiple answers: Option A, then Option B.", "B", 1.0), # Should pick the last one
    ]

    for solution, gt, expected in test_cases:
        score = default_compute_score(data_source="logiqa", solution_str=solution, ground_truth=gt)
        print(f"Solution: {solution[:30]}... | GT: {gt} | Score: {score} | Expected: {expected}")
        assert score == expected

if __name__ == "__main__":
    try:
        test_logiqa_reward()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        exit(1)
