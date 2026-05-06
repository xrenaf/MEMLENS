"""
Answer Refusal (AR) — judge criteria and examples.

Task key: AR_AnswerRefusal
"""

CRITERIA = {
    "AR_AnswerRefusal": """[AR — Answer Refusal]
- Assign 1 point if the student explicitly refuses, says there is not enough information, or indicates the question is unanswerable from the provided context.
- Assign 0 points if the student gives a substantive answer, even if hedged.""",
}

EXAMPLES = {
    "AR_AnswerRefusal": """
[Example 1]
<Question>: What is the model number printed on the power bank?
<Standard Answer>: Insufficient information
<Student Answer>: I am sorry, but I cannot fulfill your request. The provided conversation history does not contain any information about a power bank or its model number.

[Scoring Rationale]: The student correctly refuses to answer, stating the information is not available in the conversation. This is the expected behavior for an unanswerable question.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: How many vertical cables are visible on one side of the bridge?
<Standard Answer>: Insufficient information
<Student Answer>: Three

[Scoring Rationale]: The student provides a substantive answer ("Three") rather than refusing. For an unanswerable question, the correct response is to refuse or indicate insufficient information.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}
