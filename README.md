
# UCB Capstone Project
U.C. Berkeley Engineering
* Charles David Harris
* davidharrisne@gmail.com

## Predicting Progression to Alzheimer's Disease from Blood Protein Levels

### Abstract


# TODO
* Abstract of this project
  * Two approaches
---

#### Two approaches
Given the sensitiviy to false negatives, two approaches were taken


- **[`false_negative_rate.ipynb`](false_negative_rate.ipynb)** — an exhaustive search over which biomarkers matter, landing on a simple, 2-feature Logistic Regression model.

- **[`svm_pipeline.ipynb`](svm_pipeline.ipynb)** — a broader comparison using all 7 available features, landing on Support Vector Machine.

Both are deployed and combined into an ensemble in **[`main.ipynb`](main.ipynb)**, the project's main notebook. 

This report follows the CRISP-DM methodology:

1. [Business Understanding](#business-understanding)
2. [Data Understanding](#data-understanding)
3. [Data Preparation](#data-preparation)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Conclusion](#conclusion)
7. [References](#references)


# 1. Business Understanding
## Clinical Background

# TODO 
* Paint the picture of Alzhiemers
  * Define It
  * Scope
  * Symptoms
  * protei and Tau, amyloid
---
Alzheimer's disease is marked by two key proteins in the brain: **amyloid**, which forms plaques, and **tau**, which forms tangles. Drugs recently approved by the FDA remove amyloid from the brain and can slow disease progression in people with MCI or mild dementia — which makes early, accurate identification of at-risk patients clinically valuable, not just academically interesting.

> "What's exciting now is that we're looking even earlier — before symptoms begin — to see if we can predict who might be at greatest risk of developing cognitive problems in the future."
> — Clifford Jack, Jr., M.D., radiologist and lead author, [Mayo Clinic study](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/)

The Mayo Clinic's own prediction model combined age, sex, APOE genotype, and brain amyloid levels from PET scans. Of all the predictors they evaluated, amyloid levels had the single largest effect on lifetime risk of both MCI and dementia — which is why this project starts from the hypothesis that amyloid and tau levels, on their own, should carry real signal.


Worlwide, there are greater than 55 million living with Alzheimer's Disease, and baring medicl breakthroughs this number will double evey 20 years. Alzhiemrs Disease has many symptoms 

 "Accumulation of the protein beta‐amyloid outside neurons and twisted strands of the protein tau inside neurons are hallmarks. They are accompanied by the death of neurons and damage to brain tissue. Inflammation and atrophy of brain tissue are other changes."[3] 
[Mayo Clinic](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/) researchers have shown that two proteins in spinal fluid — amyloid and tau — can point toward who is at risk of developing Alzheimer's, years before symptoms appear.[4] 

This project asks the question: **Among patients with with Mild Cognitive Impairment (MCI), can we predict who will progress to Alzheimer's, using Tau and Amyloid blood/CSF protein levels?**


This project uses the Kaggle dataset [Plasma lipidomics in Alzheimer's disease](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease) — real patient measurements for 212 people, of whom 89 have a known MCI-to-Alzheimer's outcome. That is a small dataset for an epidemic-scale question, and that mismatch means that the decisions in this project were made with caution.

#### The Accuracy Trap
# TODO
* Discuss metrics
  * Precision
  * Recall
  * roc_auc   --- balance vs unbalanced
* Present strategy of analysis.
* fix the quotes
* make sure the charts match what you actually did
* Are we using F2 or recall, rox_uac?
* make a decision and stic to it

---
In a clinical setting, accuracy comes with risk. A model can post a high accuracy number while failing to identity patients who progress to alzhiemers - the False Negative diagnosis.
Acuracy must be balanced with precision, recall, F2, and AUC. 
--quote

**The business question this project answers:** given a patient already diagnosed with Mild Cognitive Impairment, can their CSF protein levels tell us whether they are likely to progress to Alzheimer's — early enough that a clinician could act on it?

**Why false negatives matter more than false positives here:** a model that says "this patient won't progress" when they actually will is a patient who doesn't get monitored or treated early. That asymmetry — missing a progressor is worse than a false alarm — shapes every modeling decision in this report, not just the final headline metric.




**In medical screening for a fatal disease, relying only on a general "False Negative Score" or a standard F1 score is not good enough because they treat false negatives and false positives with equal weight.When a disease is fatal, a False Negative means a sick patient is sent home untreated, which can lead to death. A False Positive means a healthy person gets extra tests, causing temporary anxiety but saving lives overall. Therefore, you must use metrics that specifically isolate and minimize false negatives.

The \(F_{\beta }\) Score (with β = 2): The standard F1 score weights precision and recall equally. An F₂ score adjusts the math to place twice as much importance on recall (minimizing false negatives) as it does on precision.ROC-AUC Score: This measures how well your model separates the sick population from the healthy population across all possible decision thresholds, helping you safely pick a threshold that eliminates false negatives.**

quote 
"
If you artificially balanced only the test set (e.g., undersampled negatives to evaluate), your ROC-AUC there won't reflect real-world deployment performance, since in production the disease is presumably still rare. A model can look great on a balanced test set and underperform in deployment because the operating point (threshold) that worked in balanced conditions doesn't map cleanly onto a population where positives are 1% instead of 50%.
"
### Reading a confusion matrix

Every prediction a model makes lands in one of four boxes, depending on what actually happened to the patient versus what the model guessed:

<img src="images/confusion_matrix_explainer.png" alt="Confusion matrix explainer diagram" width="380">

The box that matters most clinically is the **False Negative**: a patient who is actually heading toward Alzheimer's, but the model tells them — and their doctor — not to worry. Every metric below is really just a different way of watching that one box.

**Accuracy** — overall, how often was the model right?

$$\text{accuracy} = \frac{\text{true positives} + \text{true negatives}}{\text{everyone}}$$

Misleading on its own here, because it treats a missed progressor and a false alarm as equally bad — it can't tell them apart.

**Precision** — of everyone the model flagged as "will progress," how many actually did?

$$\text{precision} = \frac{\text{true positives}}{\text{true positives} + \text{false positives}}$$

Precision answers: of everyone the model flagged as "will progress to AD," how many actually did?

$$\text{precision} = \frac{\text{true positives}}{\text{true positives} + \text{false positives}} = \frac{\text{correct positive predictions}}{\text{all positive predictions}}$$

Recall answers: of everyone who actually progressed to Alzheimer's, how many did the model catch?

$$\text{recall} = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}} = \frac{\text{correct positive predictions}}{\text{all actual positives}}$$


F2 answers: weighing precision and recall together, but caring more about recall — how well did the model do overall?

$$F_2 = 5 \times \frac{\text{precision} \times \text{recall}}{4 \times \text{precision} + \text{recall}}$$

This is the score actually used to pick hyperparameters throughout this project, instead of accuracy — the 5/4 weighting bakes the clinical priority ("don't miss progressors") directly into the number. Sweeping every possible decision threshold shows the full precision/recall trade-off F2 is trying to summarize in one number:

<img src="images/ensemble_precision_recall.png" alt="Ensemble precision-recall curve" width="380">

AUC (area under the ROC curve) answers: how well can the model tell a progressor from a non-progressor, before you even pick a decision cutoff?

$$\text{AUC} \in [0.5,\ 1.0]$$

0.5 is a coin flip; 1.0 is perfect separation. Unlike the other three metrics, AUC doesn't depend on where the "will progress / won't progress" line gets drawn — it's a check on whether the underlying signal is real at all, before accuracy, precision, or recall (which all depend on that threshold) enter the conversation:

<img src="images/ensemble_roc.png" alt="Ensemble ROC curve" width="480">

No single metric tells the whole story on its own; accuracy alone hides exactly the failure mode that matters most here.


# 2. Data Understanding
# TODO
* Bring in the information from lipodomics
* We only have 88 customers
* This is a problem -- access to large scale clinical data not accessible
* Proceed with caution
* Acadamic excercise

---
The dataset contains 212 patients across three diagnostic categories:


<img src="images/Diagnostic.png" alt="Diagnostic counts" width="480">

Only the 89 patients already diagnosed with Mild Cognitive Impairment have a known "Progression to Alzheimer's Disease" outcome — nobody who is already healthy or already has an AD diagnosis has a progression label, because the question doesn't apply to them. So the real question this project answers is narrower than "who gets Alzheimer's": **of patients already showing MCI, who is going to get worse?** 47 of the 89 progress to Alzheimer's, 42 do not:


<img src="images/Mild_Cognitive_Impairment_.png" alt="Mild Cognitive Impairment progression counts" width="480">

**Concerns**: 89 labeled patients is a small dataset. Small samples are sensitive to exactly which patients land in a given train/test split or cross-validation fold, so results throughout this report should be read as suggestive, not definitive.

**Class balance**: at 53% Yes / 47% No, the MCI subset is close enough to balanced that this study does not apply any class-balancing technique (e.g. oversampling, class weighting) to the training data.

### Setting expectations for the biology

Before any modeling, it's worth checking visually whether the two biomarkers the Mayo Clinic hypothesis points to — CSF amyloid and CSF phosphorylated tau — actually separate progressors from non-progressors in this dataset, and in the expected direction (low amyloid, high tau → higher risk):

<img src="images/decision_boundary.png" alt="Decision boundary between progression and no progression, tau vs. amyloid" width="500">


<img src="images/pair_plot.png" alt="Pair Plot" width="500">
# 3. Data Preparation

Some MCI patients had missing values for individual biomarkers. Missing numeric values (Age, MMSE, CSF Amyloid, CSF Total tau, CSF Phosphorylated tau) were filled with that column's **median**; the one missing categorical value (APOE4 carrier status) was filled with that column's **mode** (most common value). Sex and APOE4 were converted from Yes/No text to 0/1. No rows were dropped and no class-balancing was applied — see "Class balance" above.

# 4. Modeling

With the biology sanity-checked, the next question was purely methodological: given 7 candidate features, which should the model actually use, which classification algorithm should it be, and what hyperparameters should that algorithm have? These three decisions are tangled together — the best features depend on which algorithm you're picking them for, and the best hyperparameters depend on which features and algorithm you've already chosen. There's no obviously safe order to make them in.

Rather than guess an order, this project ran two **independent, exhaustive** pipelines, each committing to a different way of resolving that tangle, and compares their results honestly rather than picking a winner in advance.

## Approach 1: Exhaustive feature search → Logistic Regression

**[`false_negative_rate.ipynb`](false_negative_rate.ipynb)** resolves the ordering problem by brute force: it checks **every possible combination of features** before ever picking an algorithm.

1. **Feature search.** Every non-empty subset of the 7 candidate features was tried — 2⁷ − 1 = **127 combinations** — each scored with a neutral, fixed Logistic Regression so the feature search wouldn't be quietly biased toward whichever algorithm came next. Every combination was filtered to keep only those with a false-negative rate at or below 20% *before* ranking by accuracy — false negatives were never allowed to be traded away for a better headline number. Only 8 of 127 combinations survived that filter. The winner: **CSF Amyloid + CSF Phosphorylated tau**, a simpler pair than initially expected.
2. **Algorithm search.** That winning pair of features was then run through seven candidate algorithms (Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN, LDA, Naive Bayes), with the same false-negative filter applied first. Logistic Regression won — a genuinely tuned win this time, not the neutral placeholder from step 1.
3. **Hyperparameter search.** `GridSearchCV` was run twice over Logistic Regression's `C` (regularization strength) and `class_weight`, once scored on plain recall and once on **F2** (a precision/recall blend that weights recall — catching progressors — twice as heavily as precision). Scoring on recall alone pushed the model toward flagging almost everyone as "will progress," which trimmed the false-negative rate a little further but gutted accuracy and precision along the way. Scoring on F2 struck a better balance and is the version actually deployed: **`C=10`, no class weighting.**

This is an honestly brute-force approach — the 127-combination search only works because there are just 7 candidate features (2ᴺ grows fast); a real biomarker panel with dozens of candidates would need a smarter search strategy instead.

## Approach 2: All 7 features → SVM

**[`svm_pipeline.ipynb`](svm_pipeline.ipynb)** takes a different route: instead of narrowing the feature list first, it uses all 7 raw features from the start and lets algorithm comparison do the work.

Seven classifiers (the same seven as above) plus a soft-voting ensemble of all seven were compared on the full feature set, first with default settings and then after `GridSearchCV` tuning (also scored on F2). Before tuning, Random Forest and SVM (RBF) came out essentially tied on recall (both caught 7 of 10 actual progressors), with Random Forest ahead on precision (0.875 vs. 0.778) and accuracy (0.778 vs. 0.722). Tuning on F2 broke that tie decisively: SVM's recall jumped to 0.90 (9 of 10 caught) while Random Forest barely moved — so **SVM (RBF)**, tuned to `C=0.1`, `gamma='scale'`, is this notebook's answer. Notably, the "wisdom of the crowd" voting ensemble of all seven models did *not* outperform the single best-tuned SVM.

## Why two different winners is not a contradiction

These two pipelines land on genuinely different answers — a 2-feature Logistic Regression vs. a 7-feature SVM — and neither is "wrong." With only 89 labeled patients, which algorithm looks best is sensitive to exactly how the data gets split, which features are kept, and which scoring function drives the search. Two reasonable, honest pipelines landing on two different winners is a real demonstration of that small-sample instability, not a bug in either one. See [`NARRATIVE.md`](NARRATIVE.md) for the full reasoning behind each decision.

Rather than pick a single "true" winner, `main.ipynb` deploys **both** models and averages their predicted probabilities as a simple two-model ensemble — treated here as one more honest data point, not a way to paper over the disagreement.

# 5. Evaluation



# 6. Conclusion




# References

1. 2025 Alzheimer's disease facts and figures. Alzheimers Dement. 2025 Apr 29;21(4):e70235. doi: 10.1002/alz.70235. PMCID: PMC12040760.
1. Mayo Clinic: [Mayo Clinic scientists create tool to predict Alzheimer's risk years before symptoms begin](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/)
1. Decision Boundaries [Analysis](decision_boundaries.ipynb)
1. Alzheimer's Disease International, https://www.alzint.org/about/dementia-facts-figures/dementia-statistics/, 
2. Dataset: [Plasma lipidomics in Alzheimer's disease](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease) (Kaggle)

