"""
Information Extraction (IE) — judge criteria and examples.

Task keys: IE_Entity, IE_PreviousInfo
"""

CRITERIA = {
    "IE_Entity": """[IE — Entity / Attribute Extraction]
This is an information extraction question about a concrete entity, count, attribute, text, object, location, or visual detail from the provided memory context.
- Assign 1 point if the student's response contains the core information from the standard answer. Minor wording differences are acceptable, but the essential information must be present and correct.
- Assign 0 points if the core information is missing, contradicted, too vague, refused, or incorrect.
- For numeric or count answers, require the exact value unless the standard answer itself is approximate.""",

    "IE_PreviousInfo": """[IE — Previous Information Extraction]
This is an information extraction question about previously mentioned information, spatial relations, counts, or attributes from earlier conversation/image context.
- Assign 1 point if the student's response recovers the same previous information as the standard answer, including the correct relation, location, count, or attribute.
- Minor wording differences are acceptable if they preserve the same meaning.
- Assign 0 points if the answer refers to the wrong previous item, gives the wrong relation/count/location, is too vague, refuses, or is unsupported.""",
}

EXAMPLES = {
    "IE_Entity": """
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

    "IE_PreviousInfo": """
[Example 1]
<Question>: Where on the sign is the photo of the mountains located?
<Standard Answer>: In the bottom right corner
<Student Answer>: The photo is in the bottom-right corner of the sign.

[Scoring Rationale]: The student's answer preserves the same location relation as the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Relative to the baking sheet, where is the salmon fillet placed?
<Standard Answer>: The right side
<Student Answer>: It is on the left side of the baking sheet.

[Scoring Rationale]: The student's answer gives the opposite spatial relation from the standard answer.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}
