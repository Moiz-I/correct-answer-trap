# Catching The Correct Answer Trap

Code and supplementary materials for:
**Catching The Correct Answer Trap: Characterising AI Tutor Blind Spots When Analysing Student Reasoning**

Accepted at AIED 2026 (27th International Conference on AI in Education)

## Overview

Students sometimes get the right answer for the wrong reason. We study how AI models handle this: they mostly don't. Fine-tuned classifiers catch around 57% of these cases. Frontier LLMs do better (84%) but generate roughly four false alarms per genuine detection in practise. The failures concentrate in specific questions where common errors happen to produce the correct numerical answer.

See `SUPPLEMENTARY.md` for prompts, model configurations, question classifications, and statistical details.

### Repository Structure

```
scripts/
  train_t5.py              T5-small LoRA fine-tuning for misconception detection
  train_bert.py             BERT-base classification baseline
  evaluation_harness.py     Per-class recall, balanced accuracy, Wilson CIs
data/
  question_classifications.json   All 15 Eedi question types with procedural/conceptual labels
SUPPLEMENTARY.md            Full prompts, configs, and statistical methods
requirements.txt            Python dependencies
```

### Dataset

We use the Eedi benchmark dataset (Rittle-Johnson et al., 2025). The dataset is publicly available through:

- [Eedi on the ACL Anthology](https://aclanthology.org/2025.aime-con-1.5/)
- [Kaggle Eedi Mining Misconceptions Competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)

Our evaluation uses a stratified 500-sample test set plus all 61 True Misconception (TM) cases from the held-out partition. The question classification in `data/question_classifications.json` maps each of the 15 unique question types used in our analysis.

#### Running the Scripts

Install dependencies:

```bash
pip install -r requirements.txt
```

**Fine-tune T5-small:**

```bash
python scripts/train_t5.py --data-dir data/processed --output-dir models/t5
```

**Fine-tune BERT:**

```bash
python scripts/train_bert.py --input_variant std-revealed
```

**Run evaluation:**

```bash
python scripts/evaluation_harness.py --predictions results.jsonl --test-set test.jsonl
```

See each script's `--help` for full options.

### Citation

```bibtex
@inproceedings{imran2026correctanswertrap,
  title={Catching The Correct Answer Trap: Characterising AI Tutor Blind Spots When Analysing Student Reasoning},
  author={Moiz Imran and Sahan Bulathwela},
  booktitle={International Conference on Artificial Intelligence in Education (AIED)},
  year={2026}
}
```

## License

MIT
