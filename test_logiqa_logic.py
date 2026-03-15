import re

def compute_score(solution_str, ground_truth):
    # Match \boxed{A} or \boxed{{B}} etc.
    matches = re.findall(r'\\boxed\{\{?([A-Za-z])\}?\}', solution_str)
    
    if not matches:
        # Try to extract C from Option(C)
        matches = re.findall(r'Option\(\s*([A-Za-z])\s*\)', solution_str)

    if not matches:
        # Try to extract C from "The answer is C."
        matches = re.findall(r'The answer is\s*([A-Za-z])\s*\.', solution_str)
        
    if not matches:
        # Try to extract C from "Answer: C"
        matches = re.findall(r'Answer:\s*([A-Za-z])\s*', solution_str)
        
    if not matches:
        # Try to extract C from "C is the correct answer"
        matches = re.findall(r'([A-Za-z])\s*is the correct answer', solution_str)
        
    if not matches:
        # Try to extract C from "The correct answer is C"
        matches = re.findall(r'The correct answer is\s*([A-Za-z])\s*', solution_str)
        
    if not matches:
        # Try to extract C from Option C
        matches = re.findall(r'Option\s*([A-Za-z])\s', solution_str)
        
    if not matches:
        # Try to extract A from **Option (A)**
        matches = re.findall(r'Option\s*\(\s*([A-Za-z])\s*\)', solution_str)

    if matches:
        extracted_answer = [letter.upper() for letter in matches]
        if extracted_answer[-1] == ground_truth.upper():
            return 1.0
        else:
            return 0.0
    else:
        return 0.0

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
        score = compute_score(solution_str=solution, ground_truth=gt)
        print(f"Solution: {solution[:30]}... | GT: {gt} | Score: {score} | Expected: {expected}")
        assert score == expected

if __name__ == "__main__":
    try:
        test_logiqa_reward()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        exit(1)
