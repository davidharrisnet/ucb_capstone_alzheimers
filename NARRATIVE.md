# The Story Behind This Project

*A personal narrative to accompany [README.md](README.md) and the analysis notebooks — the "why" and the "what happened along the way," not just the results.*

## Why I picked this problem

Alzheimer's isn't a niche interest for me — it's an epidemic, and I wanted to work on something that mattered. The Mayo Clinic claim that pulled me in is specific: that CSF tau and amyloid levels can point toward who is going to progress to Alzheimer's before the symptoms do. That's a real, testable prediction problem, and I wanted to see if it held up.

Real clinical data for something like this is protected, for good reason, so I couldn't get it. What I could get was Kaggle's plasma lipidomics dataset — real patient data, not synthetic, but a small set: 212 patients, only 89 with a known MCI progression outcome. That mismatch — a big, epidemic-scale question chased with a small dataset — is the tension that runs through this whole project. The data is jumpy in a way a larger cohort wouldn't be, and I don't think it's safe to assume results here scale cleanly to a bigger population. Every caveat later in this narrative traces back to that.

## The scope narrowed faster than I expected

I started out assuming I'd predict Alzheimer's diagnosis broadly across all 213 patients. The data had other plans: the "Progression to Alzheimer's Disease" label only exists for the 89 patients already diagnosed with Mild Cognitive Impairment — nobody already healthy ("Control") or already diagnosed with AD has a progression outcome, because the question doesn't apply to them. So the real question became narrower and, I think, more useful: *of patients already showing MCI, who is going to get worse?* That reframing shaped everything downstream.

## Testing the clinical hypothesis first

Before touching a classifier, I checked whether the biomarkers the literature points to — CSF Amyloid, CSF Total tau, CSF Phosphorylated tau, APOE4, MMSE — actually carried signal in *this* dataset, using single-feature logistic regression p-values and AUC. Age and Sex washed out (p > 0.35), but Amyloid, Total tau, and APOE4 all cleared p < 0.05, which was reassuring — it meant the dataset wasn't too small or noisy to reproduce a known clinical relationship before I asked it to do anything harder.

## Which to tune first?

Once the clinical hypothesis held up, the next problem wasn't about the data anymore — it was about method. There were three interlocking decisions to make: which features to use, which classification algorithm to use, and how to tune that algorithm's hyperparameters. There's no clean order to make them in. Tune hyperparameters first and you're tuning them for the wrong feature set. Pick features first and you're picking them for whichever algorithm happens to be running at the time — which might not even be the best algorithm once tuned. Pick the algorithm first, based on default-parameter performance, and you might throw out an algorithm that would have won handily once given the chance to tune. Each decision depends on the other two, so there's no obviously safe first move.

The honest answer to "which do I do first" turned out to be: don't guess the order — make each stage exhaustive on its own terms, and put the false-negative concern first at every stage rather than treating it as a tiebreaker at the end. Concretely, that became three passes: first, search every feature subset with a single fixed algorithm (plain Logistic Regression) as a neutral evaluator, so the feature choice wasn't biased by which algorithm happened to like which features. Second, take the winning feature subset and compare it across seven different algorithms. Third, take the winning algorithm on the winning features and grid-search its hyperparameters. At every single one of those three stages, the first thing I did was filter out anything with a false-negative rate above 20% — *then* rank what survived by accuracy, AUC, or F2. Accuracy never got to overrule a false-negative problem; it only got to break ties among candidates that had already cleared that bar.

Concretely (see `main.ipynb`), that meant looping `itertools.combinations` over every subset size from 1 feature up to all 7, building the full non-empty power set — 2⁷ − 1 = 127 combinations — and scoring each one. That's the honest limitation of this approach: the power set grows as 2ᴺ, so it only works because the feature list is small. Add a handful more candidate biomarkers and an exhaustive search like this stops being feasible; it would need to become a smarter search (greedy forward selection, regularization, or similar) instead of brute force. For this study, with 7 candidate features, brute force was affordable and worth the honesty it bought.

## The twist I didn't expect

Early on (see the model comparison table in the README), Gradient Boosting and Random Forest overtook Logistic Regression once hyperparameters were tuned — "not what I expected," and a little uncomfortable, because tree ensembles are harder to explain to a clinician than a two-coefficient logistic model.

That discomfort is why I went back and did the search properly instead of trusting the first answer. Stage one: an exhaustive check of all 127 non-empty feature subsets, scored with plain Logistic Regression as a neutral stand-in so the feature search wasn't quietly favoring whichever algorithm I'd compare next, cross-validated with `StratifiedKFold`, filtered down to only the combinations keeping the false-negative rate at or below 20%. Only 8 of 127 survived that filter. The winner wasn't the Amyloid + Total tau pair I'd settled on earlier — it was **Amyloid + Phosphorylated tau**. Stage two: I re-ran that winning feature pair through seven algorithms (Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN, LDA, Naive Bayes), same false-negative filter first — and Logistic Regression won again, this time for real, not as the neutral placeholder from stage one. Stage three: grid-searching Logistic Regression's own hyperparameters (`C`, `class_weight`) scored by F2 (which weights recall over precision) instead of accuracy landed on `C=10`, no class weighting. Simpler model, better-justified answer at every stage, and it took a lot more discipline in the search to get there than the earlier round did.

## The accuracy trap

Early on, accuracy looked like the obvious scoreboard — highest accuracy wins. But accuracy is a trap in a clinical setting like this one: with only 89 labeled patients and a real class imbalance, a model can post a good accuracy number while quietly missing the patients who matter most — the ones actually progressing to Alzheimer's. A false negative here isn't a rounding error; it's a patient who doesn't get monitored or treated early because the model told them they were fine.

That's why I stopped treating accuracy as the headline metric and leaned on it alongside precision, recall, F2, and AUC instead of in place of them. Precision and recall expose the tradeoff directly — how many flagged patients are true progressors, versus how many actual progressors get caught. F2 scores that tradeoff in a single number while explicitly weighting recall over precision, which matches the clinical priority: missing a progressor is worse than over-flagging someone who turns out fine. AUC is the one I trust as a threshold-independent check — it tells me whether a feature or model has genuine separating power at all, before I let accuracy, precision, or recall numbers (which all depend on where you set the decision threshold) into the conversation. No single metric tells the whole story on its own; accuracy alone hides exactly the failure mode that matters most here.

## Reading the confusion matrix

Every prediction the model makes lands in one of four boxes, depending on what actually happened to the patient versus what the model guessed:

| | Model predicted: **will progress** | Model predicted: **won't progress** |
|---|---|---|
| **Actually progressed** | True Positive — caught it | **False Negative — missed it** |
| **Actually did not progress** | False Positive — false alarm | True Negative — correctly cleared |

The box that matters most clinically is the **False Negative**: a patient who is actually heading toward Alzheimer's, but the model tells them — and their doctor — not to worry. Every metric below is really just a different way of watching that one box.

**Accuracy** answers: overall, how often was the model right?

$$\text{accuracy} = \frac{\text{true positives} + \text{true negatives}}{\text{everyone}}$$

Misleading on its own here, because it treats a missed progressor and a false alarm as equally bad — it can't tell them apart.

**Precision** answers: of everyone the model flagged as "will progress to AD," how many actually did?

$$\text{precision} = \frac{\text{true positives}}{\text{true positives} + \text{false positives}} = \frac{\text{correct positive predictions}}{\text{all positive predictions}}$$

High precision means few false alarms, but it says nothing about how many progressors slipped through.

**Recall** (a.k.a. sensitivity) answers: of everyone who actually progressed to Alzheimer's, how many did the model catch?

$$\text{recall} = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}} = \frac{\text{correct positive predictions}}{\text{all actual positives}}$$

This one speaks directly to the False Negative box — it drops every time the model misses a real progressor.

**F2 score** answers: weighing precision and recall together, but caring more about recall — how well did the model do?

$$F_2 = 5 \times \frac{\text{precision} \times \text{recall}}{4 \times \text{precision} + \text{recall}}$$

This is the score I actually optimized for instead of accuracy — the "5" and "4" weighting bakes the clinical priority ("don't miss progressors") directly into the number, instead of treating precision and recall as equally important.

**AUC** (area under the ROC curve) answers: how well can the model tell a progressor from a non-progressor, before you even pick a cutoff?

$$\text{AUC} \in [0.5, 1.0]$$

0.5 means no better than a coin flip; 1.0 means perfect separation. Unlike the other four, AUC doesn't depend on where the "will progress / won't progress" line gets drawn — it's a check on whether the underlying signal is real at all, independent of any particular threshold.

## Tuning the winner: back to the accuracy trap

With Logistic Regression picked as the algorithm and Amyloid + Phosphorylated tau picked as the features, the last open question was hyperparameters: what value of `C` (regularization strength), and whether to weight the classes. This is exactly where "The accuracy trap" stops being an abstract warning and turns into an actual decision `GridSearchCV` has to make — it needs one scoring function to rank candidate hyperparameters against each other, and whichever one you hand it is the one it optimizes for. So I ran the same modest grid (`C` in [0.01, 0.1, 1, 10, 100], `class_weight` in [None, 'balanced']) twice: once scored on plain recall, once scored on F2.

Scoring on recall alone picked `C=0.01` and did shave the false-negative count slightly — 7 missed progressors (FNR 14.9%) versus 8 missed (FNR 17.0%) under F2 — but it got there by gutting accuracy (59.6%) and precision (0.58): predicting "will progress" so liberally that catching almost everyone came at the cost of a model that barely discriminates. Scoring on F2 picked `C=10` instead, and landed at 74.2% accuracy and 0.72 precision while still keeping the false-negative rate comfortably under the 20% bar. Both hyperparameter sets technically minimize false negatives well enough; F2 is the one that also keeps accuracy from collapsing. (Notably, AUC barely moved between the two — 0.777 vs. 0.777 — confirming the underlying separating power of the features didn't change, only where the decision threshold got drawn, which is exactly what "Reading the confusion matrix" says AUC is supposed to tell you.) F2 won not because it cares less about false negatives than pure recall does, but because it refuses to buy a marginal drop in false negatives at the price of a model that's barely better than a coin flip on everything else.

## Sitting with the false negatives

The recurring discomfort throughout was false negatives: a patient the model tells "you're not progressing" who actually is. That's not an abstract metric — it's a person who doesn't get monitored or treated early. That's the whole reason the final search filtered on false-negative rate before it ever looked at accuracy, and why F2 (not accuracy, not plain AUC) is the scoring function that picked the final hyperparameters (`C=10`, no class weighting). The final holdout test (18 patients — 10 actual progressors, 8 non-progressors, none of them touched during any of the search) came back at 77.8% accuracy, 0.8 precision, and 0.85 AUC. In confusion-matrix terms: 6 true negatives, 2 false positives, 2 false negatives, 8 true positives — 8 of 10 actual progressors correctly caught, missing only 2. That's a false-negative rate of exactly 20%, landing right on the threshold I'd used to filter every stage of the search. Not perfect, but a deliberate, defensible tradeoff rather than an accident of the metric I happened to optimize — and, honestly, a genuinely good result to land on for a two-feature model built from 89 patients.

## A different notebook, a different winner

`svm_pipeline.ipynb` — the second of the two training-approach notebooks — asks a related but not identical question. Instead of narrowing to a winning pair of biomarkers first, it runs all seven features through seven classifiers plus a soft-voting ensemble, straight out of the box, on a single 80/20 stratified split (71 train / 18 test). Before any hyperparameter tuning, Random Forest and SVM (RBF) come out essentially tied on recall — both catch 7 of 10 actual progressors — with Random Forest ahead on precision (0.875 vs. 0.778) and accuracy (0.778 vs. 0.722). Once `GridSearchCV` is pointed at F2 instead of accuracy, that tie breaks: SVM's recall jumps to 0.90 (9 of 10 progressors caught, F2 0.849) while Random Forest barely moves — so SVM (RBF), not Logistic Regression, is that notebook's final answer.

I ran the same final-holdout printout used in `main.ipynb` against this notebook's tuned SVM, on the identical 18-patient test set (10 progressors, 8 non-progressors). It came back: 4 true negatives, 4 false positives, 1 false negative, 9 true positives — 72.2% accuracy, 0.69 precision, 0.73 AUC, and a false-negative rate of just 10%. Set side by side with `main.ipynb`'s Logistic Regression result (8/10 caught, FNR 20%, 77.8% accuracy, 0.80 precision), the SVM catches one more progressor but pays for it with twice as many false alarms (4 vs. 2) and a meaningfully lower accuracy. Neither result is "more correct" — they're two different points on the same precision/recall tradeoff from "The accuracy trap," produced by two pipelines that ended up weighting that tradeoff a little differently.

That's a genuinely different winner than `main.ipynb`'s pipeline reaches, and I don't think that's a contradiction so much as the small-sample caveat showing up in practice instead of just being asserted. With 89 patients total, which algorithm looks best is sensitive to exactly how the data gets split, which features are kept in, and which scoring function drives the grid search. Two reasonable pipelines land on two different "winning" algorithms, and both are defensible given their own setup — which is exactly why I wouldn't take either one's specific algorithm choice as gospel without a bigger, external validation set.

## What I'd want before trusting this clinically

89 labeled patients is not a lot. Every choice — stratified folds instead of a single split, a modest hyperparameter grid instead of a wide one, filtering on false-negative rate before optimizing anything else — was a hedge against overfitting to a small sample. I saved the final pipeline (`alzheimers_progression_model.joblib`) so the exact model is reproducible, but I wouldn't call this deployment-ready without a larger, external validation cohort first.

*(Your close: what's next for you — another dataset, a different modeling angle, or is this the end of the capstone story? Worth a few sentences here.)*
