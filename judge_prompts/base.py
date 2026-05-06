"""
Base judge prompt template — shared across all question types.

Contains universal grading rules, universal examples for problematic outputs
(circular reasoning, no-answer traces), and placeholders for type-specific
criteria and examples.
"""

QA_BENCHMARK_JUDGE_PROMPT = """Now your role is a grading teacher. Your task is to review and score student answers based on reference standard answers for a question-answering benchmark. You need to notice the following key points:
- First, extract the final answer from the student's solution, then analyze and judge whether the answer is correct.
- Scoring should only refer to the final answer obtained by the student; there is no need to examine whether the intermediate problem-solving steps are correct.
- If the response contains both hesitation and a clear answer, judge the answer itself.
- If the response contains both a refusal and a guessed answer, judge the final committed answer.
- If the response gives multiple inconsistent answers, assign 0 points.
- If the student's response shows circular reasoning — repeatedly revisiting the same evidence, flip-flopping between answers (e.g., "yes... wait no... actually yes... let me reconsider... no"), with no clear final commitment — assign 0 points. A correct answer mentioned during reasoning does NOT count unless it is the clearly stated final conclusion.
- Only the student's LAST clearly stated answer counts. If the student initially says "yes" but later concludes "no" (or vice versa), score based on the LAST position only. Intermediate answers during reasoning are not final answers.
- If the student's output is entirely a reasoning trace (e.g., "let me check... scanning evidence... re-reading session 5...") with no clear concluding answer statement, assign 0 points — even if the correct answer appears somewhere within the reasoning.
- If the student's output begins with "[...truncated earlier reasoning...]", this means only the final portion of a longer response is shown. Focus on extracting the answer from this final portion.

Below are examples of problematic student outputs that apply to ALL task types. Study these before grading.

[Universal Example A — Circular Reasoning -> 0 points]
<Question>: Is the cat in Mark and Jenny's 5-year anniversary post named Mittens?
<Standard Answer>: Yes
<Student Answer>: let me check session 5 mark has cat named mittens session 7 mark and jenny anniversary post there is no link wait let me re-read session 5 says mittens session 7 says anniversary actually looking at this again wait is it possible let me reconsider mark has mittens in session 5 the post is about couple therefore answer is likely no wait let me double check session 5 mark has cat mittens actually maybe yes but text doesnt say
[Scoring Rationale]: The output flip-flops between "yes" and "no" multiple times with no clear final commitment. The last position is ambiguous ("maybe yes but text doesnt say"). This is circular reasoning.
In summary, the student's answer deserves 0 points.
[JSON]: {{"answer_score": 0}}

[Universal Example B — Redundant But Committed -> Score Normally]
<Question>: Is the trailing plant in the cafe safe for my kitten?
<Standard Answer>: No
<Student Answer>: the plant looks like pothos pothos is toxic to cats therefore answer is no let me double check is pothos safe for cats no is answer no yes wait let me verify image 3 is pothos image 9 is pothos pothos is toxic therefore answer must be no
[Scoring Rationale]: Despite excessive self-verification, the student consistently commits to "no" throughout and never wavers to a different answer. The final answer is clearly "No", which matches the standard answer.
In summary, the student's answer deserves 1 point.
[JSON]: {{"answer_score": 1}}

[Universal Example C — Reasoning Trace With No Answer -> 0 points]
<Question>: How many sessions mentioned the coffee shop?
<Standard Answer>: 3
<Student Answer>: scanning session 1 yes coffee mentioned session 2 no session 3 yes coffee again session 4 no session 5 maybe let me re-read session 5 it mentions cafe is that same as coffee shop need to check session 6 no session 7 unclear let me look at session 5 again
[Scoring Rationale]: The student walks through sessions but never states a final count. The output is entirely a reasoning trace with no concluding answer.
In summary, the student's answer deserves 0 points.
[JSON]: {{"answer_score": 0}}

Now proceed with grading. Remember: the universal examples above apply regardless of task type.

- When analyzing and judging whether the answer is correct, you need to write down the scoring rationale, organize it into clear statements that follow the logical flow. The summary of the scoring rationale should be placed at the end, using the following format: "In summary, the student's answer deserves x points" (where x represents the student's specific score).
- Keep the whole process concise, within 150 words.
- Provide the score based on your analysis and display it in a code block in "JSON" format.
- An item is covered if it is strictly mentioned or unambiguously implied by a semantic equivalence. This includes numerical equivalence (e.g., 10% and 0.1), synonyms (e.g., UK and United Kingdom), plural/singular forms (e.g., "apple" and "apples"), and equivalent date formats (e.g., 2024-01-15 and January 15, 2024). However, do not accept loosely related concepts.
- Ignore minor formatting differences, capitalization, punctuation, and equivalent wording when meaning is unchanged.
Your output format is:
[Scoring Rationale]:
[Score]: x points
[JSON]:
{{"answer_score": <integer_value>}}

Below is the grading rubric:
[Scores]:
The scoring scale consists of 2 levels in total, from highest to lowest: 1 point, 0 points (the minimum is 0 points).
[Tier Details]:
1 point: Assign 1 point if the student's final answer matches the standard answer under the task-specific criteria below.
0 points: Assign 0 points if the student's final answer does not match the standard answer, the student refuses to answer, claims insufficient information (except for AR tasks), or does not clearly answer.

[Task-Specific Criteria]:
{task_criteria}

{examples}
"""
