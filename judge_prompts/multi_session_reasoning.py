"""
Multi-Session Reasoning (MSR) — judge criteria and examples.

Task keys: MSR_YesNo, MSR_Counting, MSR_Arithmetic
"""

CRITERIA = {
    "MSR_YesNo": """[MSR — Yes/No (Entity Resolution)]
This is a yes/no question.
- Assign 1 point if the student's final answer is semantically equivalent to the standard answer (e.g., "Yes" matches "Yes", "No" matches "No"), even if phrased differently (e.g., "That's correct" for "Yes").
- Assign 0 points if the answer means the opposite, refuses, or does not clearly answer.""",

    "MSR_Counting": """[MSR — Counting]
This is a counting question. The standard answer is a specific number.
- Assign 1 point if the student clearly states the exact number in any equivalent form (e.g., "3", "three", "3.0").
- Assign 0 points if the student gives a different number, a range, an approximation, refuses, or does not clearly answer.""",

    "MSR_Arithmetic": """[MSR — Arithmetic]
This is an arithmetic question. The standard answer is a specific value.
- Assign 1 point if the student gives the same value in an equivalent format (e.g., "0.5" and "1/2" and "50%").
- Assign 0 points if the student gives a different value, an unsupported approximation, refuses, or does not clearly answer.""",
}

EXAMPLES = {
    "MSR_YesNo": """
[Example 1]
<Question>: Is the cat in Mark and Jenny's 5-year anniversary post named Mittens?
<Standard Answer>: Yes
<Student Answer>: Yes, the cat in Mark and Jenny's 5-year anniversary post is named Mittens.

[Scoring Rationale]: The student's final answer is "Yes", which matches the standard answer "Yes". The additional explanation does not contradict the answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Did the cat named after a crypto creator knock over the phone displaying the Binance alert?
<Standard Answer>: Yes
<Student Answer>: The text does not explicitly state whether the cat knocked over the phone. It only mentions the cat was sitting next to the phone when it crashed.

[Scoring Rationale]: The student does not give a clear "Yes". Instead, the student hedges and effectively refuses to commit to an answer. The standard answer is "Yes".
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",

    "MSR_Counting": "",

    "MSR_Arithmetic": """
[Example 1]
<Question>: How much total have I spent on coffee makers?
<Standard Answer>: $260.00
<Student Answer>: Based on the conversation, you spent $110 on the first coffee maker and $150 on the second one, totaling $260.

[Scoring Rationale]: The student's final answer is $260, which is numerically equivalent to the standard answer $260.00.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: How much total have I spent on coffee makers?
<Standard Answer>: $260.00
<Student Answer>: I do not have access to your personal financial data, including your coffee maker purchases. I cannot provide you with a total amount spent on coffee makers.

[Scoring Rationale]: The student refuses to answer and claims insufficient information, but this is not an AR task. The information was available in the conversation context.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}
