# Experiments

## Problem

The goal is to decide whether each prior radiology examination should be shown to the radiologist while reading the current examination. Each request contains one or more patient cases, and each case includes:

- one current study
- a list of prior studies for the same patient

The API must return exactly one prediction for every prior study. Missing predictions count as incorrect, so the endpoint performs bulk inference over the full request and always returns one output per prior examination.

## Data and Validation Setup

The public split contains labeled examples of prior examinations and whether each prior is relevant to the current study. I used this public split for local experimentation and validation.

To reduce leakage, I evaluated models using grouped validation splits by `case_id`. This ensures that prior examinations from the same case do not appear in both training and validation.

The final model was trained on the full public split before packaging and deployment.

## Baselines

### All-false baseline

The first baseline returned `false` for every prior examination. Because the dataset is imbalanced, this can achieve a reasonable baseline accuracy, but it misses clinically relevant priors and has very low recall.

### Exact-match baseline

The second baseline returned `true` only when the current and prior study descriptions exactly matched. This improved precision but missed many related studies with slightly different descriptions, abbreviations, modality wording, or contrast wording.

Examples of near-duplicate descriptions that exact matching may miss include:

- `CT HEAD WITHOUT CONTRAST`
- `CT HEAD WITHOUT CNTRST`
- `MRI BRAIN W WO CONTRAST`
- `MRI BRAIN WITH AND WITHOUT CONTRAST`

## Modeling Approach

The submission uses a deterministic scikit-learn Logistic Regression model. I chose this approach because it is fast, explainable, easy to deploy, and reliable under the challenge timeout constraints.

The model uses a combination of text features and engineered clinical-style features.

## Feature Engineering

### Text features

The model uses TF-IDF features over:

```text
current_study_description [SEP] prior_study_description
```

I tested both word-level and character-level TF-IDF features.

Final text features:

- Word TF-IDF n-grams from 1 to 3
- Character TF-IDF n-grams from 3 to 5

Character n-grams helped with abbreviation and spelling variations such as:

- `CNTRST` vs `CONTRAST`
- `WO` vs `WITHOUT`
- `MR` vs `MRI`

### Basic engineered features

The initial engineered features included:

- exact description match
- modality match, such as CT-to-CT or MRI-to-MRI
- token overlap count
- Jaccard similarity between current and prior descriptions
- date gap between current and prior exams
- binary date-window indicators for 1, 2, and 3 years

### Body-region features

I added rule-based body-region extraction from study descriptions. The motivation was that radiology prior relevance depends heavily on whether the prior is anatomically comparable to the current exam.

The body-region extractor maps study descriptions into broad groups such as:

- brain/head
- neck
- chest
- cardiac/vascular
- abdomen/pelvis
- spine
- breast
- upper extremity
- lower extremity
- whole body

Body-region features included:

- same body region
- known same body region
- same modality and same body region

This produced the largest validation improvement in the project.

### Rank and recency features

I then added case-level rank and recency features so the model could reason about each prior in the context of the full patient history, rather than treating each prior independently.

Rank/recency features included:

- prior rank by recency within the case
- number of newer prior examinations
- whether the prior was the most recent prior overall
- whether the prior was the most recent prior with the same modality
- whether the prior was the most recent prior with the same body region
- whether the prior was the most recent prior with both same modality and same body region
- number of newer priors with the same modality
- number of newer priors with the same body region
- number of newer priors with both same modality and same body region

This was motivated by the idea that radiologists often prefer recent comparable priors, and older priors may be less relevant when newer exams of the same anatomy or modality exist.

### Contrast and procedure compatibility features

I also tested contrast/procedure compatibility features to capture differences such as:

- `WITH CONTRAST`
- `WITHOUT CONTRAST`
- `W WO CONTRAST`
- `CTA`
- `MRA`
- portable exams
- limited/screening exams
- laterality, such as left/right/bilateral

Tested features included:

- same contrast status
- known same contrast status
- both exams angiographic
- same angiographic status
- same laterality
- known same laterality
- left/right laterality conflict
- same limited/screening status
- same portable status

This experiment improved grouped validation accuracy, but it reduced the public smoke-test score compared with the rank/recency model. Because the challenge includes a public smoke test and final hidden scoring, I did not select this as the final submission model.

## Experiment Results

| Experiment | Threshold | Validation Accuracy | Public Smoke Test | Notes |
|---|---:|---:|---:|---|
| Word TF-IDF + basic engineered features | 0.45 | 0.8940 | 96.53% | Initial logistic regression model |
| Word + character TF-IDF | 0.46 | 0.9000 | 96.53% | Improved abbreviation robustness |
| Body-region features | 0.44 / 0.54 tested | 0.9402 | 97.69% | Largest validation improvement |
| Body-region + rank/recency features | 0.39 | 0.9402 | 98.27% | Final selected model |
| Body-region + rank/recency + contrast/procedure features | 0.59 | 0.9434 | 97.69% | Higher validation, lower smoke score |

## Body-Region Model Comparison

After adding body-region features, I compared multiple Logistic Regression settings.

| Model | Threshold | Validation Accuracy | Precision | Recall | F1 | Public Smoke Test |
|---|---:|---:|---:|---:|---:|---:|
| C=1.0 conservative | 0.54 | 0.9402 | 0.9397 | 0.8076 | 0.8687 | 97.11% |
| C=2.0 higher-recall | 0.44 | 0.9402 | 0.8980 | 0.8527 | 0.8747 | 97.69% |

The C=2.0 / threshold=0.44 body-region model had stronger F1 and a better public smoke-test score at that stage.

## Multi-Split Stability Check

To reduce the risk of choosing a model that only worked on one validation split, I evaluated the two strongest body-region models across five grouped random splits.

| Model | Mean Accuracy | Std Accuracy | Mean Precision | Mean Recall | Mean F1 | Public Smoke Test |
|---|---:|---:|---:|---:|---:|---:|
| C=2.0, threshold=0.44 | 0.9281 | 0.0082 | 0.8833 | 0.8148 | 0.8475 | 97.69% |
| C=1.0, threshold=0.54 | 0.9250 | 0.0105 | 0.9141 | 0.7662 | 0.8335 | 97.11% |

The C=2.0 / threshold=0.44 body-region model was stronger across grouped validation splits. Later, adding rank/recency features improved the public smoke-test result further.

## Rank and Recency Results

Adding rank/recency features preserved grouped validation accuracy while improving public smoke-test performance.

Results:

- Previous body-region model validation accuracy: `0.9402`
- Body-region + rank/recency validation accuracy: `0.9402`
- Body-region + rank/recency validation confusion matrix: `[[4379, 145], [213, 1253]]`
- Best threshold after rank/recency features: `0.39`
- Public smoke-test accuracy improved from `97.69%` to `98.27%`

This became the final selected model because it kept strong grouped validation accuracy while producing the best public smoke-test score observed during experimentation.

## Contrast and Procedure Results

Adding contrast/procedure compatibility features improved validation accuracy, but reduced the public smoke-test score compared with the rank/recency model.

Results:

- Previous best body-region + rank/recency validation accuracy: `0.9402`
- Contrast/procedure feature model validation accuracy: `0.9434`
- Best threshold: `0.59`
- Validation precision: `0.9519`
- Validation recall: `0.8097`
- Validation F1: `0.8750`
- Validation confusion matrix: `[[4464, 60], [279, 1187]]`
- Public smoke-test accuracy: `97.69%`

I treated this as a useful experiment, but did not select it as the final model because the validation improvement was modest and the public smoke-test score dropped from `98.27%` to `97.69%`.

## Final Selected Model

The final selected model uses:

- Logistic Regression
- Word TF-IDF n-grams from 1 to 3
- Character TF-IDF n-grams from 3 to 5
- Basic engineered similarity/date features
- Modality matching
- Body-region matching
- Case-level rank and recency features
- Decision threshold: `0.39`
- Logistic Regression regularization: default `C=1.0`

I selected this model because it achieved strong grouped validation accuracy and the best public smoke-test score observed during experimentation.

Final observed metrics:

- Grouped validation accuracy: `0.9402`
- Validation confusion matrix: `[[4379, 145], [213, 1253]]`
- Public smoke-test accuracy: `98.27%`

## API Behavior

The endpoint performs bulk inference over all cases and all prior studies in one request. It logs:

- request ID
- number of cases
- number of prior studies

The endpoint avoids one model call per prior and instead performs vectorized inference over all rows. This keeps the API fast enough for evaluator requests.

## What Worked

The strongest improvements came from adding domain-inspired features:

1. Character n-grams improved robustness to abbreviations and spelling variations.
2. Body-region extraction helped identify anatomically comparable priors.
3. Rank and recency features helped model the importance of recent comparable studies.
4. Grouped validation by `case_id` helped avoid overestimating performance from leakage across prior exams from the same case.

## What Was Weaker

- The all-false baseline missed too many relevant priors.
- Exact-description-only rules had high precision but low recall.
- Handwritten rules alone were brittle because radiology descriptions contain many abbreviations and near-duplicates.
- Hyperparameter tuning alone produced only small gains compared with feature engineering.
- Contrast/procedure compatibility improved validation accuracy but reduced the public smoke-test score, so it was not selected for the final model.

## Next Improvements

Given more time, I would explore:

1. More robust radiology normalization, including a larger abbreviation dictionary.
2. A hierarchy-aware body-region matcher, such as abdomen/pelvis compatibility and spine subregion compatibility.
3. Modality-specific calibration, since prior relevance rules differ for CT, MRI, ultrasound, x-ray, and mammography.
4. Error analysis on the remaining public smoke-test mistakes.
5. A small ensemble or gradient-boosted model over the engineered features, while keeping the current logistic regression as a fast baseline.
