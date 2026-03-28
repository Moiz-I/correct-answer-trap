# Supplementary Materials

**Paper:** Catching The Correct Answer Trap: Characterising AI Tutor Blind Spots When Analysing Student Reasoning

## Prompt Details

### Task-Specific Prompt (used for Gemini and Llama)

```
You are a maths tutor analysing student work for misconceptions.

CRITICAL: A student can have a misconception even with a correct
answer. Focus on their REASONING.

MISCONCEPTION EXISTS when:
1. Describes patterns without connecting to underlying concept
2. States a rule that contradicts their work

Example: Uses absolute values then "makes it negative" for integer subtraction (e.g., "8-5=3 so -3") - fails on (-5)-(-8)

NO MISCONCEPTION when:
1. Valid procedure even if informal
2. Correct intuition even if imprecise
3. Method would work correctly on similar problems

Question: {question_text}
Correct Answer: {correct_answer}
Student Answer: {student_answer}
Student Explanation: {student_explanation}

Diagnosis:
```

### Model Configurations

| Model | API/Source | Model ID | Temperature | Notes |
|-------|-----------|----------|-------------|-------|
| Gemini 3 Flash | Google AI | `gemini-3-flash-preview-04-17` | 0 | Tested with low (1024 token) and high (8192 token) thinking budgets |
| Llama-3.3-70B | Groq | `llama-3.3-70b-versatile` | 0 | |
| Llama-3.1-8B | Groq | `llama-3.1-8b-instant` | 0 | |
| T5-small | Fine-tuned | `t5-small` | N/A | LoRA fine-tuning (r=8, alpha=16) on classification task |
| BERT-base | Fine-tuned | `bert-base-uncased` | N/A | Standard classification head, included as architecture robustness check |

All models accessed March 2026.

## Question Types

The Eedi dataset contains 15 unique question types. Following Hiebert and Lefevre (1986), we classified each as procedural (P), conceptual (C), or mixed (M) before analysis. As discussed in RQ2, the vulnerability to the correct answer trap is item-specific rather than category-wide.

| Question | Classification | TM Count |
|----------|---------------|----------|
| Integer subtraction: (-8) - (-5) | P | 138 |
| Equivalent fractions: A/10 = 9/15 | P | 107 |
| Fraction division: 1/2 divided by 6 | P | 19 |
| Fraction multiplication: 2/3 times 5 | P | 5 |
| Fraction addition: 1/3 + 2/5 | P | 3 |
| Linear equation: 2y = 24 | P | 1 |
| Fraction of quantity: 3/8 of 24 | C | 10 |
| Fraction of shape (visual) | C | 9 |
| Probability: P = 0.9, describe | C | 7 |
| Inverse proportion: 3 people, 192 hrs | C | 6 |
| Polygon angles: 144 degrees | C | 5 |
| Comparing decimals | C | 16 |
| Fraction of fraction: 2/3 then 1/3 | C | 4 |
| Fraction of quantity: 3/5 of 120 | C | 0 |
| Dot patterns (sequence) | M | 17 |

## Statistical Analysis

We use two statistical tests throughout the paper.

**Fisher's exact test** compares proportions between independent groups. We use this for:

- Sensitivity analysis (RQ2): Comparing TM rates between procedural and conceptual questions, and testing whether the effect persists when the two highest-concentration items are excluded (odds ratio collapses from 5.6 to 1.0, p = 0.48)
- PRM800K validation (Discussion): Comparing detection rates between correct-answer and wrong-answer cases

**McNemar's test** compares paired classifiers evaluated on the same samples. We use this for:

- RQ3: Comparing T5 vs Gemini on the same 61 TM test cases
- Prompt validation: Comparing task-specific vs PedCoT prompts on the same samples

McNemar's test is appropriate here because the models are evaluated on identical cases, making the observations paired rather than independent.

**Confidence intervals** use the Wilson score method, which provides accurate coverage for proportions even with small sample sizes (unlike the normal approximation, which can produce impossible intervals near 0 or 1).

We use p < 0.05 as the significance threshold throughout.
