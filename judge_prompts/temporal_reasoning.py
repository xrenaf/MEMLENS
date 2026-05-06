"""
Temporal Reasoning (TR) — judge criteria and examples.

Task keys: TR_DurationComparison, TR_OrderRanking, TR_DateExtraction
"""

CRITERIA = {
    "TR_DurationComparison": """[TR — Duration Comparison (A/B)]
This is a binary choice question. The student must select one of two options.
- Assign 1 point if the student clearly selects the same option as the standard answer, regardless of phrasing.
- Assign 0 points if the student selects the other option, refuses, or does not make a clear selection.""",

    "TR_OrderRanking": """[TR — Order Ranking]
This is an ordering question. The standard answer is an exact sequence of items.
- Assign 1 point ONLY if the student gives the exact same sequence in the same order.
- Assign 0 points if any item is missing, added, duplicated, misplaced, or if the order is unclear.""",

    "TR_DateExtraction": """[TR — Date Extraction]
This is a date extraction question. The standard answer is a specific date.
- Assign 1 point if the student gives the same date in any unambiguous equivalent format (e.g., "Jan 15, 2024" and "2024-01-15" and "15/01/2024").
- Assign 0 points if the date is different, incomplete when a full date is required, ambiguous, or missing.""",
}

EXAMPLES = {
    "TR_DurationComparison": """
[Example 1]
<Question>: Which of the following two durations is longer? Duration 1: The time I spent on Miller Hall. Duration 2: The time I spent on Music Festival Job. A. Duration 1 is longer. B. Duration 2 is longer.
<Standard Answer>: B
<Student Answer>: B. Duration 2 is longer.

[Scoring Rationale]: The student clearly selects "B", which matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Which of the following two durations is longer? Duration 1: The time I spent with the NutriPeak Nutrition. Duration 2: The time I spent with the BlendIt Pro Blender. A. Duration 1 is longer. B. Duration 2 is longer.
<Standard Answer>: A
<Student Answer>: B. Duration 2 is longer.

[Scoring Rationale]: The student selects "B", but the standard answer is "A". The selection is opposite.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",

    "TR_OrderRanking": """
[Example 1]
<Question>: Rank the following cities by population from largest to smallest: Tokyo, Delhi, Shanghai.
<Standard Answer>: Tokyo, Delhi, Shanghai
<Student Answer>: 1. Tokyo 2. Delhi 3. Shanghai

[Scoring Rationale]: The student's sequence is Tokyo, Delhi, Shanghai, which exactly matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Rank events A, B, C in chronological order.
<Standard Answer>: B, A, C
<Student Answer>: A, B, C

[Scoring Rationale]: The student's sequence is A, B, C, but the standard answer is B, A, C. The order does not match.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",

    "TR_DateExtraction": """
[Example 1]
<Question>: When did I complete the Scandinavian design course at Copenhagen Design Academy?
<Standard Answer>: 2024/08/31
<Student Answer>: You completed the Scandinavian design course on August 31st, 2024.

[Scoring Rationale]: The student's answer "August 31st, 2024" is an unambiguous equivalent of the standard answer "2024/08/31".
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: When did I visit Marutama Ra-men to taste their chicken broth?
<Standard Answer>: 2023/01/20
<Student Answer>: January 20, 2021

[Scoring Rationale]: The student's answer is "January 20, 2021", but the standard answer is "2023/01/20" — the year is wrong (2021 vs 2023).
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}
