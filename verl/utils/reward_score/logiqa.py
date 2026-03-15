# -*- coding: utf-8 -*-

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
