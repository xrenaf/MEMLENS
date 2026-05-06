"""
Judge prompt registry — collects type-specific criteria and examples,
provides build_judge_prompt() and get_task_key().
"""

from .base import QA_BENCHMARK_JUDGE_PROMPT
from .information_extraction import CRITERIA as IE_CRITERIA, EXAMPLES as IE_EXAMPLES
from .knowledge_update import CRITERIA as KU_CRITERIA, EXAMPLES as KU_EXAMPLES
from .temporal_reasoning import CRITERIA as TR_CRITERIA, EXAMPLES as TR_EXAMPLES
from .multi_session_reasoning import CRITERIA as MSR_CRITERIA, EXAMPLES as MSR_EXAMPLES
from .answer_refusal import CRITERIA as AR_CRITERIA, EXAMPLES as AR_EXAMPLES

# Merged registries (9 task keys total)
TASK_CRITERIA = {**IE_CRITERIA, **KU_CRITERIA, **TR_CRITERIA, **MSR_CRITERIA, **AR_CRITERIA}
TASK_EXAMPLES = {**IE_EXAMPLES, **KU_EXAMPLES, **TR_EXAMPLES, **MSR_EXAMPLES, **AR_EXAMPLES}


def get_task_key(question_type: str, question_subtype: str, reference: str) -> str:
    """Map (question_type, question_subtype, reference) to a task key."""
    if question_type == "multi_session_reasoning":
        if question_subtype == "arithmetic":
            return "MSR_Arithmetic"
        elif question_subtype == "counting":
            return "MSR_Counting"
        elif question_subtype == "entity_resolution":
            return "MSR_YesNo" if reference.strip() in ("Yes", "No") else "MSR_Counting"
        return "MSR_Counting"
    elif question_type == "temporal_reasoning":
        if question_subtype == "duration_comparison":
            return "TR_DurationComparison"
        elif question_subtype == "order_ranking":
            return "TR_OrderRanking"
        elif question_subtype == "temporal_info_extraction":
            return "TR_DateExtraction"
        return "TR_DateExtraction"
    elif question_type == "knowledge_update":
        return "KU_KnowledgeUpdate"
    elif question_type == "answer_refusal":
        return "AR_AnswerRefusal"
    return "IE_InformationExtraction"


def build_judge_prompt(task_type: str, question: str, reference: str,
                       prediction: str, old_answer: str = None) -> str:
    """
    Build the full judge prompt for a given task type.

    Args:
        task_type: One of the keys in TASK_CRITERIA
        question: The question text
        reference: The standard/ground-truth answer
        prediction: The student/model answer
        old_answer: (Optional) For KU tasks, the outdated old answer
    """
    criteria = TASK_CRITERIA.get(task_type, "")
    examples = TASK_EXAMPLES.get(task_type, "")

    system_prompt = QA_BENCHMARK_JUDGE_PROMPT.format(
        task_criteria=criteria,
        examples=examples,
    )

    if task_type == "KU_KnowledgeUpdate" and old_answer:
        current_case = f"""[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Old (Outdated) Answer>: {old_answer}
<Student Answer>: {prediction}

"""
    elif task_type == "AR_AnswerRefusal":
        current_case = f"""[Current Case]
<Question>: {question}
<Standard Answer>: Insufficient information
<Student Answer>: {prediction}

"""
    else:
        current_case = f"""[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Student Answer>: {prediction}

"""

    return system_prompt + current_case + "[Scoring Rationale]:"
