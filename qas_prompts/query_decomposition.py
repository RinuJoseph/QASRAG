

PROMPT_TEMPLATE = """\
You are an expert at decomposing complex multi-hop questions into a sequence
of simple, sub-questions that can be answered one at a time.

## Task
Given a complex question, break it down into the minimum number of ordered
sub-questions such that:
1. Each sub-question asks for ONE specific piece of information.
2. Each sub-question is self-contained and answerable independently (given
   the answers to previous sub-questions).
3. Later sub-questions may reference the answers of earlier ones using
   natural language placeholders like  "the answer to Q2".

## Rules
- Do NOT skip reasoning steps; each hop in the question becomes its own
  sub-question.
- Do NOT merge two facts into one sub-question.
- Keep sub-questions short, clear, and in natural language.
- Preserve the logical order: the output of one sub-question feeds into
  the next.
- Do not answer the sub-questions — only decompose.

## Output Format
Original Question: <restate the input question>

Sub-questions:
Q1: <first sub question>
Q2: <next question, referencing Q1's answer if needed>
Q3: <next question, referencing previous answers if needed>
...
Qn: <last sub-question>

## Now decompose the following question:
Original Question: "{question}\""""


def build_messages(question: str):
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]
