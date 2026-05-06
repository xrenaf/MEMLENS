"""
Information Extraction (IE) — judge criteria and examples.

Task key: IE_InformationExtraction
"""

CRITERIA = {
    "IE_InformationExtraction": """[IE — Information Extraction]
This is an information extraction question.
- Assign 1 point if the student's response contains the core information from the standard answer. Minor wording differences are acceptable, but the essential information must be present and correct.
- Assign 0 points if the core information is missing, contradicted, too vague, refused, or incorrect.""",
}

EXAMPLES = {
    "IE_InformationExtraction": """
[Example 1]
<Question>: What specific word is painted in large white letters on the pavement in the middle lanes?
<Standard Answer>: SCHOOL
<Student Answer>: SCHOOL

[Scoring Rationale]: The student's answer is "SCHOOL", which exactly matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Where on the sign is the photo of the mountains located?
<Standard Answer>: In the bottom right corner
<Student Answer>: The photo of the mountains is located in the upper right corner of the sign.

[Scoring Rationale]: The student says "upper right corner" but the standard answer is "bottom right corner". The location is incorrect — "upper" vs "bottom" is a substantive difference.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}
