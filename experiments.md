# Experiments

## Problem

The goal is to decide whether each prior radiology examination should be shown to the radiologist while reading the current examination. Each prediction is made for a pair of:

- current study description/date
- prior study description/date

The API must return exactly one prediction per prior study.

## Baseline

The first baseline was to return `false` for every prior. Because the public split is imbalanced, this gives a reasonable baseline but misses clinically relevant priors.

A second baseline was to return `true` only when the prior study description exactly matched the current study description. This improves precision but misses related exams that are not exact text matches.

## Final model

The final submission uses a deterministic scikit-learn logistic regression model. Features include:

- TF-IDF word n-grams over `current_description [SEP] prior_description`
- exact-description match
- modality match, such as CT-to-CT, MRI-to-MRI, XR-to-XR
- token overlap and Jaccard similarity between descriptions
- date gap features between current and prior exams
- binary date-window indicators for one, two, and three years

The model is trained on the full public split before packaging. During local validation, I used a group split by `case_id` so that prior exams from the same case did not leak across train and validation.

## Local validation

Using an 80/20 grouped validation split by `case_id`, the selected model achieved about 0.89 validation accuracy. A probability threshold of 0.45 worked slightly better than the default 0.50 on the validation split.

## API behavior

The endpoint performs bulk inference over all cases and all prior studies in one pass. It logs:

- request ID
- number of cases
- number of prior studies

This helps debug evaluator calls and avoids making one prediction call per prior.

## What worked

- Text similarity between current and prior study descriptions was the strongest signal.
- Exact description match was very predictive.
- Modality match helped because radiologists often need same-modality priors for comparison.
- Date gap features helped reduce over-selection of very old unrelated exams.

## What failed or was weaker

- A pure all-false baseline was too conservative.
- Exact-description-only rules had high precision but low recall.
- Handwritten rules were brittle because radiology descriptions have many abbreviations and near-duplicate variants.

## Next improvements

Given more time, I would add:

1. A stronger domain-specific normalization layer for radiology abbreviations, such as `CNTRST` to `CONTRAST`.
2. Body-part extraction so that anatomically related studies are grouped more reliably.
3. Calibration by modality and body region, since relevance rules differ for chest CT, brain MRI, mammography, and x-ray follow-ups.
4. A small gradient boosted tree or ensemble combining the text model with engineered clinical features.
5. Error analysis on false negatives, because skipped relevant priors are costly in the challenge scoring.

## Final Model Update

After the initial baseline, I added character n-gram TF-IDF features in addition to word-level TF-IDF and engineered metadata features. This helped the model handle abbreviation and spelling variations in radiology study descriptions, such as `CNTRST` versus `CONTRAST`, and `WO` versus `WITHOUT`.

The final selected model uses:
- Word TF-IDF n-grams: 1 to 3
- Character TF-IDF n-grams: 3 to 5
- Engineered features: exact description match, modality match, Jaccard token overlap, token intersection count, and date-gap features
- Logistic Regression with `C=1.0`
- Decision threshold: `0.46`

Validation results:
- Baseline validation accuracy: `0.8940`
- Character n-gram model validation accuracy: `0.9000`
- Best validation threshold: `0.46`
- Public smoke-test accuracy: `96.53%`

I also tested additional regularization tuning. A model with `C=0.5` and threshold `0.43` improved grouped validation accuracy to `0.9030`, but reduced the public smoke-test score from `96.53%` to `94.80%`. I selected the `C=1.0`, threshold `0.46` model as the final submission because it provided the best balance between validation performance and smoke-test stability.

## Body-Region Feature Experiment

After the character n-gram experiment, I added rule-based body-region extraction from study descriptions. The motivation was that radiology relevance often depends not only on modality and text overlap, but also on whether the prior exam is anatomically comparable to the current exam.

The body-region extractor maps study descriptions into broad regions such as:
- Brain/head
- Neck
- Chest
- Cardiac/vascular
- Abdomen/pelvis
- Spine
- Breast
- Upper extremity
- Lower extremity
- Whole body

New engineered features included:
- Same body region
- Known same body region
- Same modality and same body region

This produced the largest validation improvement in the project.

Results:
- Word TF-IDF + engineered baseline: `0.8940` grouped validation accuracy
- Word + character n-gram TF-IDF: `0.9000` grouped validation accuracy
- Word + character n-gram TF-IDF + body-region features: `0.9402` grouped validation accuracy

I also compared multiple logistic regression settings:

| Model | Threshold | Validation Accuracy | Precision | Recall | F1 | Public Smoke Test |
|---|---:|---:|---:|---:|---:|---:|
| C=1.0 conservative | 0.54 | 0.9402 | 0.9397 | 0.8076 | 0.8687 | 97.11% |
| C=2.0 higher-recall | 0.44 | 0.9402 | 0.8980 | 0.8527 | 0.8747 | 97.69% |

The final candidate model was selected based on grouped validation accuracy, public smoke-test performance, and the balance between precision and recall. The C=2.0 / threshold=0.44 model had the strongest F1 and the best public smoke-test score, while tying for best validation accuracy.

### Multi-Split Stability Check

To reduce the risk of choosing a model that only performed well on one validation split, I evaluated the two strongest body-region models across five grouped random splits.

| Model | Mean Accuracy | Std Accuracy | Mean Precision | Mean Recall | Mean F1 | Public Smoke Test |
|---|---:|---:|---:|---:|---:|---:|
| C=2.0, threshold=0.44 | 0.9281 | 0.0082 | 0.8833 | 0.8148 | 0.8475 | 97.69% |
| C=1.0, threshold=0.54 | 0.9250 | 0.0105 | 0.9141 | 0.7662 | 0.8335 | 97.11% |

The C=2.0 / threshold=0.44 model was selected because it had the stronger average accuracy and F1 across grouped validation splits, while also achieving the best public smoke-test score.

## Rank-Based Recency Feature Experiment

After adding body-region features, I added case-level rank and recency features so the model could reason about each prior examination in the context of the full patient history, rather than treating each prior independently.

New rank/recency features included:
- Prior rank by recency within the case
- Number of newer prior examinations
- Whether the prior was the most recent prior overall
- Whether the prior was the most recent prior with the same modality
- Whether the prior was the most recent prior with the same body region
- Whether the prior was the most recent prior with both same modality and same body region
- Number of newer priors with the same modality
- Number of newer priors with the same body region
- Number of newer priors with both same modality and same body region

This feature family was motivated by the observation that radiologists often prefer recent comparable priors, and that older priors may be less relevant when newer studies of the same anatomy or modality exist.

Results:
- Body-region model validation accuracy: `0.9402`
- Body-region + rank/recency model validation accuracy: `0.9402`
- Body-region + rank/recency model validation confusion matrix: `[[4379, 145], [213, 1253]]`
- Best threshold after rank/recency features: `0.39`
- Public smoke-test accuracy improved from `97.69%` to `98.27%`

The final selected model uses word TF-IDF, character n-gram TF-IDF, engineered modality/body-region features, and case-level rank/recency features. I selected this version because it preserved grouped validation accuracy while improving recall/F1 balance and achieving the best public smoke-test score.

## Contrast and Procedure Compatibility Experiment

I added contrast/procedure compatibility features to capture differences such as `WITH CONTRAST`, `WITHOUT CONTRAST`, `W WO CONTRAST`, `CTA`, `MRA`, portable exams, limited/screening exams, and laterality.

New features included:
- Same contrast status
- Known same contrast status
- Both exams angiographic
- Same angiographic status
- Same laterality
- Known same laterality
- Left/right laterality conflict
- Same limited/screening status
- Same portable status

Results:
- Previous best body-region + rank/recency model validation accuracy: `0.9402`
- Contrast/procedure feature model validation accuracy: `0.9434`
- Best threshold: `0.59`
- Validation confusion matrix: `[[4464, 60], [279, 1187]]`
- Public smoke-test accuracy remained `98.27%`

I kept this version because it improved grouped validation accuracy while preserving the best public smoke-test score.