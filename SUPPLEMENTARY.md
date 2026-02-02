# Supplementary Materials

**Paper:** The Correct Answer Trap: Characterising AI Tutor Blind Spots in Student Feedback

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

| Model          | API/Source               | Temperature | Notes                                     |
| -------------- | ------------------------ | ----------- | ----------------------------------------- |
| Gemini 3 Flash | `gemini-3-flash-preview` | 0           | Tested with low and high thinking budgets |
| Llama-3.3-70B  | Groq                     | 0           |                                           |
| Llama-3.1-8B   | Groq                     | 0           |                                           |
| T5-base        | Fine-tuned               | N/A         | LoRA fine-tuning on classification task   |

## Question Type Classifications

Following Hiebert and Lefevre (1986), we classified each Eedi question type as procedural, conceptual, or mixed before analysis.

**Procedural (6 types):** Questions where a memorisable rule can produce correct answers without understanding.

- Integer subtraction (e.g., (-8) - (-5))
- Equivalent fractions (e.g., A/10 = 9/15)
- Fraction division (e.g., 1/2 ÷ 6)
- [Additional types in dataset]

**Conceptual (8 types):** Questions requiring understanding that rote rules cannot shortcut.

- Fraction-of-shape (visual interpretation)
- Comparing decimals
- [Additional types in dataset]

**Mixed (1 type):**

- Dot patterns (sequence recognition)

## Statistical Analysis

We use two statistical tests throughout the paper.

**Fisher's exact test** compares proportions between independent groups. We use this for:

- RQ2: Comparing True_Misconception rates between procedural and conceptual questions
- RQ4: Comparing detection rates between correct-answer and wrong-answer cases in PRM800K

We report odds ratios to quantify effect size. An odds ratio of 5.6 (RQ2) means a response to a procedural question is 5.6 times more likely to be a True_Misconception case than a response to a conceptual question.

**McNemar's test** compares paired classifiers evaluated on the same samples. We use this for:

- RQ3: Comparing T5 vs Gemini on the same 61 True_Misconception test cases
- Prompt ablation: Comparing task-specific vs PedCoT prompts on the same samples

McNemar's test is appropriate here because the models are evaluated on identical cases, making the observations paired rather than independent.

**Confidence intervals** use the Wilson score method, which provides accurate coverage for proportions even with small sample sizes (unlike the normal approximation, which can produce impossible intervals near 0 or 1).

We use $p < 0.05$ as the significance threshold throughout.
