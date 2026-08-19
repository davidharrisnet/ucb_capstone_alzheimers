<div align="center">

# Capstone Project

<div align="center">

### Predicting Progression to Alzheimer's Disease from Blood Protein Levels
---


<div align="left">

## 1. Problem Statement

### Alzhiemer's Disease Epidemic

Worlwide, there are greater than 55 million living with Alzheimer's Disease, and baring medical breakthroughs this number will double every 20 years[1]. Within medical research there is a consensus that "Accumulation of the protein beta‐amyloid outside neurons and twisted strands of the protein tau inside neurons are hallmarks. They are accompanied by the death of neurons and damage to brain tissue. Inflammation and atrophy of brain tissue are other changes[2]." There is excitement and active research of identifying signs of alzhiemer's before it is manifest, and finding ways to treat the brain's amylid plaques and tau tangles.

<a id="business-understanding"></a>
##### Business Understanding
This project explored a Kaggle dataset to test the hypothesis that Amyloid and Tau protein levels are predictive of someone developing alzhiemers in the future. Medical diagnosis are concerned for false negatives, so these models were evaluated by Recall, and ROC_UAC This project asks the question: Among patients with with Mild Cognitive Impairment (MCI), can we predict who will progress to Alzheimer's, using Tau and Amyloid blood/CSF protein levels?


This project explored a Kaggle dataset to test the hypothesis that Amyloid and Tau protein levels are predictive of someone developing alzhiemers in the future. Medical diagnosis are concerned for false negatives, so these models were evaluated by Recall, and ROC_UAC 
This project asks the question: **Among patients with with Mild Cognitive Impairment (MCI), can we predict who will progress to Alzheimer's, using Tau and Amyloid blood/CSF protein levels?**



## 2 Model Outomes and Predicitions

Supervised Classification problem degrees of dementia,but alzhiermls is a binary diagnosos Yes/No. 

#### Clinical Metrics

In a clinical setting, accuracy comes with risk. A model can post a high accuracy number while failing to identity patients who progress to alzhiemers - the false negative diagnosis. Likewise counting the false negative rate is insufficient because if ignores patients who are falsey diagnosed - false positive - and have to undergo the stress and financial burden of uneccassary treatment. 


### Reading a confusion matrix

Every prediction a model makes lands in one of four boxes, depending on what actually happened to the patient versus what the model guessed:

<img src="images/confusion_matrix_explainer.png" alt="Confusion matrix explainer diagram" width="380">

**Accuracy** — overall, how often was the model right?

$$\text{accuracy} = \frac{\text{true positives} + \text{true negatives}}{\text{everyone}}$$

The can be misleading because it treats a missed progressor and a false alarm as equally bad — it can't tell them apart.

**Recall** answers: of everyone who actually progressed to Alzheimer's, how many did the model catch?

$$\text{recall} = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}} = \frac{\text{correct positive predictions}}{\text{all actual positives}}$$

This measure the ability of a model to catch all true positive cases, but ignores the cost of false positives. 

**F2** answers: weighing precision and recall together, but caring more about recall. It has a focus on false negatives while penalizing mised diagnosis - false positives. 

$$F_2 = 5 \times \frac{\text{precision} \times \text{recall}}{4 \times \text{precision} + \text{recall}}$$

For this work, F2, was chosen as the metric to look for. F2 is a balance between precission and recall placing twice the weight of recall as precision. 


## 3. Data Aquisition

## [Plama lipidomic in Alzheimer's diseasd](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease)
### Data Card
### About Datset

**Introduction**
Alzheimer's disease (AD) is a progressive neurodegenerative disorder that affects humans. It is typically characterized by cognitive impairment, which affects speech, behavior, and visual orientation. As cognitive capabilities decline, daily activities become more challenging, disabilities are experienced, and death occurs. Alzheimer's disease (AD) is strongly associated with abnormal lipid metabolism.
This dataset contains 213 plasma samples, including 20 controls, 89 samples from individuals with mild cognitive impairment, and 104 samples from individuals with Alzheimer's disease. Furthermore, the dataset includes information on age, sex, cognitive evaluation results, and cerebrospinal fluid biomarkers indicative of Alzheimer's disease.

**Authors**
Gerard Piñol-Ripoll, Farida Dakterzada, Joaquim Sol, Mariona Jové, Reinald Pamplona

**- Creative Commons License of the dataset:**
Attribution-NonCommercial-NoDerivatives (BY-NC-ND)

**- Dataset Digital Object Identifier (DOI):**
https://doi.org/10.34810/data614

**- Publications related to the dataset:**
Dakterzada, F., Jové, M., Huerto, R., Carnes, A., Sol, J., Pamplona, R., & Piñol-Ripoll, G. (2023). Changes in Plasma Neutral and Ether-Linked Lipids Are Associated with The Pathology and Progression of Alzheimer’s Disease. Aging and Disease, 14(5), 1728.
## Clinical Background




#### 
<a id="data-preparation"></a>
## 4. Data Preprocessing/Preparation

<a id="data_understanding"></a>
#### Data Understanding (CRISPR)

Preprocessing cleaning
#### Setting expectations for the biology


<a id="data-preparation"></a>
# 3. Data Preparation
Each of the juyter notebooks begin with the following code block.
It is commented here to explain each step.

Load data
```
import pandas as pd
df = pd.read_csv('data/plasma_lipidomics.csv')
```
Only use patients with a value in the "Progression to Alzheimer's Disease" column
```
mci = df[df["Progression to Alzheimer's Disease"].notna()].copy()
```

Isolate the numeric columns
```
numeric_cols = ['Age', 'MMSE', 'CSF Amyloid (pg/mL)', 'CSF Total tau (pg/mL)',
                 'CSF Phosphorylated tau (pg/mL)']
```
Fill nan values with the median value of the column
```
for col in numeric_cols:
    median_val = mci[col].median()
    mci[col] = mci[col].fillna(median_val)
```
APOE4 is a "Yes" or "No" value
Fill missing values with the most common value
'''
mode_val = mci['APOE4'].mode()[0]
mci['APOE4'] = mci['APOE4'].fillna(mode_val)
'''

Convert categorical text columns to numeric (0/1)
There were no nan values for Sex
```
mci['Sex'] = (mci['Sex'] == 'Male').astype(int)           # Male=1, Female=0
mci['APOE4'] = (mci['APOE4'] == 'Yes').astype(int)        # carries APOE4 allele=1, no=0
```

Set the Target
```
mci['Target'] = (mci["Progression to Alzheimer's Disease"] == 'Yes').astype(int)
```
Create the featuer set by appending numeric cols with Sex and APOE4
```
feature_cols = numeric_cols + ['Sex', 'APOE4']
```
Set up the X and y values for model evaluation and traning for the rest of the excercise
```
X = mci[feature_cols]
y = mci['Target']
```
### Target 
For the Diagnostic column only patients with a 'Mild Cognigitive" profile have a value of "Yes" or "No" in the "Progression to Alzheimer's" colummn. This subset then comprised the rows for this study. 
```
overlap_yes = df[(df['Diagnostic'] == 'Mild Cognitive Impairment') & (df["Progression to Alzheimer's Disease"] == 'Yes')]
overlap_no  = df[(df['Diagnostic'] == 'Mild Cognitive Impairment') & (df["Progression to Alzheimer's Disease"] == 'No')]

print("Non-MCI patients with Progression = Yes:", len(overlap_yes))
print("Non-MCI patients with Progression = No:", len(overlap_no))
```
Non-MCI patients with Progression = Yes: 47

Non-MCI patients with Progression = No: 42

Only the 89 patients already diagnosed with Mild Cognitive Impairment have a known "Progression to Alzheimer's Disease" So question for this project was **of patients already showing Mild Cognivite Impairment, who is going to develop alzhiemer's**

<img src="images/Mild_Cognitive_Impairment_.png" alt="Mild Cognitive Impairment progression counts" width="380">

**Concerns**: 89 labeled patients is a small dataset. Small samples are sensitive to exactly which patients land in a given train/test split or cross-validation fold, so results throughout this report should be read as suggestive, not definitive.


**Class balance**: at 53% Yes / 47% No, the MCI subset is close enough to balanced that this study does not apply any class-balancing techniques.

<a id="modeling"></a>
## 5. Modeling
Before any modeling, it's worth checking visually whether the two biomarkers the Mayo Clinic hypothesis points to — CSF amyloid and CSF phosphorylated tau — actually separate progressors from non-progressors in this dataset, and in the expected direction (low amyloid, high tau → higher risk):

<img src="images/decision_boundary.png" alt="Decision boundary between progression and no progression, tau vs. amyloid" width="500">

This test was approached from two approaches:

## Approach 1: Exhaustive feature search → Logistic Regression






<a id="model-evaluation"></a>
## 6. Model Evaluation
Show the charts

#### [KNearestNeighbors](knearest_neighbors_pipeline.ipynb)

| Metric | Value |
|---|---|
| f2 | 0.8000 |
| Recall | 0.7000 |
| Accuracy | 0.6667 |
| Precision | 0.7000 |
| ROC AUC | 0.6125 |
| False Negatives | 3 |
| False Negative Rate | 0.3000 |



 LogisticRegression arose as the best model and was deployed for testing in  [Logisticregression](logistic_regression_pipeline.ipynb) 

| Profile | CSF Amyloid (pg/mL) | CSF Phosphorylated tau (pg/mL) |
|---|---|---|
| High-risk profile | 368.2 | 119.2 |
| Median profile | 604.0 | 63.5 |
| Low-risk profile | 1062.0 | 37.1 |
| Extreme AD-like | 257.0 | 512.0 |
| Extreme low-risk | 1845.0 | 22.4 |


| Feature | Importance | Std |
|---|---|---|
| CSF Total tau (pg/mL) | 0.216667 | 0.067814 |
| CSF Amyloid (pg/mL) | 0.161111 | 0.094444 |
| Sex | 0.155556 | 0.095581 |
| Age | 0.133333 | 0.066667 |
| CSF Phosphorylated tau (pg/mL) | 0.111111 | 0.065734 |
| APOE4 | 0.111111 | 0.049690 |
| MMSE | 0.055556 | 0.043033 |

[knearest_neighbors_pipeline](knearest_neighbors_pipeline.ipynb#coefficients)

<a id="conclusion"></a>
# 6. Conclusion

This was as good excercise, but the data set was small. 
In previous iterations of this project I read other datasets. In one it appeared that in a set of 22,000 (sythetically created) data, Age was the greatest indecator of alzeimers'. The narrow focus on developing alzeimers in the future is a worthwile endeafgor. 
Recent studies exploring breathing carbon diocide reduction in Amyloid levels.  Imagine a cheap therapy breathing carbon dioxide.  A long wank in the park? Rr

# Next Steps



# Report Artifacts
1. This README has a high level overview of the project. 
2. [main.ipynb](main.ipynb) contain more granular work for **Business Understanding** and **Data Understanding** 
3. [decison_boundaries](decision_boundaries.ipynb) contains exploartory charts for "Progression" and "No Progresion" 
4. [logistic_regression_pipeling](logistic_regression_pipeline.ipynb) contains the work for the "Parameter First" approach
5. [knearest_neighbors_pipelikne](knearest_neighbors_pipeline) has code for the "Classifiction First" approach

<a id="references"></a>
# References

1. 2025 Alzheimer's disease facts and figures. Alzheimers Dement. 2025 Apr 29;21(4):e70235. doi: 10.1002/alz.70235. PMCID: PMC12040760.
1. Mayo Clinic: [Mayo Clinic scientists create tool to predict Alzheimer's risk years before symptoms begin](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/)
1. Decision Boundaries [Analysis](decision_boundaries.ipynb)
1. Alzheimer's Disease International, https://www.alzint.org/about/dementia-facts-figures/dementia-statistics/, 
2. Dataset: [Plasma lipidomics in Alzheimer's disease](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease) (Kaggle)
1.  Glodzik L, Randall C, Rusinek H, de Leon MJ. Cerebrovascular reactivity to carbon dioxide in Alzheimer's disease. J Alzheimers Dis. 2013;35(3):427-40. doi: 10.3233/JAD-122011. PMID: 23478306; PMCID: PMC3776495. [National Library of Medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC3776495/)
