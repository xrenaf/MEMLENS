"""
Knowledge Update (KU) — judge criteria and examples.

Task key: KU_KnowledgeUpdate
"""

CRITERIA = {
    "KU_KnowledgeUpdate": """[KU — Knowledge Update]
This is a knowledge update question testing whether the model tracks the latest state.
The correct current answer and the outdated old answer are both provided.
- Assign 1 point if the student gives the current (correct) answer.
- Assign 0 points if the student gives the outdated old answer, any other incorrect answer, refuses, or does not clearly answer.""",
}

EXAMPLES = {
    "KU_KnowledgeUpdate": """
[Example 1]
<Question>: What do I treat as my must-have now that I plan my meals around it?
<Standard Answer>: brioche
<Old (Outdated) Answer>: sourdough
<Student Answer>: You now consider brioche your must-have bread that you plan meals around.

[Scoring Rationale]: The student's answer is "brioche", which matches the current standard answer, not the outdated answer "sourdough".
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: What do I treat as my must-have now that I plan my meals around it?
<Standard Answer>: brioche
<Old (Outdated) Answer>: sourdough
<Student Answer>: A versatile, high-quality protein source like chicken, beans, or tofu is a great starting point for building a week's worth of meals.

[Scoring Rationale]: The student's answer mentions generic protein sources, which is neither the current answer "brioche" nor the outdated answer "sourdough". The answer is simply incorrect.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}
