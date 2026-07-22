# UCB Capstone Project 
## Alzheimers
U.C. Berkeley Engineering Data Science Project


## Introducion
The dataset for this study comes from the Kaggle [Plasma lipidomics in Alzheimer's disease](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease). It has 212 rows of real patient data with demographic data and the key protiens 'CSF Amyloid (pg/mL)', and 'CSF Total tau (pg/mL)'	along with the predictive meature 'Progression to Alzheimer's Disease'. This dataset was chosen to test a claim by the [Mayo Clinic](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/) that the presence of amyloid and tau in the brain is predictive of developing alzeimer's 'plaques' and 'tangles' in the future. If true, this is a promising develoment in combating Alzeimers by detecting the signs with a PET scan. 


## Data Analysis

There are the categories for diagnosis.
<p align="center">
  <img src="images/Diagnostic.png" alt="Centered Logo" width="300">
</p>

Upon investigation, only patients with "Mild Cognitive Impairment" have values for "Progression to Alzheimers". The "Progression to Alzheimers" the Control group and patients already diagnosed with "Alzeimer's" have Nan values. 

<p align="center">
  <img src="images/Mild_Cognitive_Impairment_.png" alt="Centered Logo" width="300">
</p>


<p align="center">
  <img src="images/Progression_to_Alzheimer's_Disease.png" alt="Centered Logo" width="300">
</p>

So the Progression to Alzheimer's Disease and "Mild Cognitive Impariment" are the same  89 patients. 
Therefore for this study I only considerd "Mild Cognitive Impairment" and "Progression to Alzheimer's Disease".

## The Study ##

Focusing on the Mayo Clinic's hypothesis, CSF Phosphorylated tau (pg/mL), CSF Total tau (pg/mL), and CSF Amyloid (pg/mL) were analyzed. The first phase was to look at combinations of the parameters with with the same model StandardScaler/LogistitRegression. For the second phase, once the best parameters are determined, looked at the best classifier method.


## Logistic Regression

| X values | Accuracy | Area Under the Curve (UAC) |
|---|---|---|
| APOE4 | 0.7191 | 0.6694 |
| CSF Amyloid (pg/mL) | 0.71910 | 0.751013 |
| CSF Total tau (pg/mL) | 0.71910 | 0.7272 |
| CSF Phosphorylated tau (pg/mL)| 0.6404 | 0.7155 |
| 'CSF Phosphorylated tau (pg/mL)', 'CSF Total tau (pg/mL)' | 0.719101 | 0.72061 |
| **'CSF Amyloid (pg/mL)','CSF Total tau (pg/mL)'**| **0.730337** | **0.801165** |


## Classification Models for CSF Amyloid (pg/mL)','CSF Total tau (pg/mL)
| Model | Accuracy | AUC | Recall
|---|---|---|--|
| **Logistic Regression** | 0.730 | **0.801** | 0.766 |
| SVM (RBF) | 0.730 | 0.780 | 0.766 |
| LDA | 0.730 | 0.802| 0.766 |
|Naive Bayes | 0.742| 0.799 | 0.766 |
| **Gradient Boosting** | **0.775** | 0.799 | 0.745 |
| Random Forest | 0.730 | 0.772 | 0.723 |
| KNN | 0.708 | 0.754 | 0.681 |





**Conclusion:** Looking at the seven classifier pipelines, Gradient Boosting has the highest accuracy but Logistic Regression has the highest Areas Under the Curve and a higher Recall. For medical dianosis, false negatives are important so there are no missed diagnosises. Logistic Regression, then appears to be the best choice.  This is advantageous as Logitic Regression is very interpretable, and is  more efficient algorithm than Gradient Boosting. 

Also of note, it appears the model choice matters far less than feature choice did. Going from single-feature amyloid (UAC 0.75 )and single-feature tau (0.72) to combined amyloid+tau (0.80) was a much bigger jump than switching from Logistic Regression. It is reassuring that the biology is doing more work than the algorithm.
