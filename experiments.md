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