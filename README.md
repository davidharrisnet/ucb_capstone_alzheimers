# Note this is is active development, and will stabalize by 9 AM August 15th.

# UCB Capstone Project 
U.C. Berkeley Engineering Data Science Project
## Protien predictors for developing Alzheimers



### Kaggle Plasma lipidomics in Alzheimer's disease

[Kaggle Plasma lipidomics in Alzheimer's disease](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease)


## Synopsis 
The potential of developing of Alzheimer's disease can be predicted by two key proteins in the brain - amyloid and tau. 
[[Mayo Clinic]](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/)
This report used the Kaggle dataset [Plasma lipidomics in Alzheimer's disease](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease), which contains critical Tau and Amyloid measures of 212 real patient data to predict 'Progression to Alzheimer's Disease'.

The detailed analysis for this project is contained in the jupyter notebook [Jupyter Notebook  Lipidomics](lipidomics_notebook.ipynb).
The outline folows the steps of the CRISP-DM methodology, and contains the following sections:

1. Introduction
2. Data Analysis
3. Validated amyloid and tau are the best predicitive parameters of the dataset with Logistic Regression
4. Determine the best classification algorithm 'Logistic Regression','Random Forest', 'SVM (RBF)','Gradient Boosting','K-Nearest Neighbors','LDA''Naive Bayes'
5. Investigate Hyper Parameters ([hyperparamemter notebook](hyperparameters.ipynb)]
   
6. Minimizing False Negatives
7. Conclusion


#### Data Card

##### About Datset

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




   

# Business Understanding
## Clinical Background

[Mayo Clinic](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/)

Predicting Alzheimer's disease
Alzheimer's disease is marked by two key proteins in the brain: amyloid, which forms plaques, and tau, which forms tangles. Drugs recently approved by the Food and Drug Administration remove amyloid from the brain and can slow the rate of disease progression for people with MCI or mild dementia.

Photo of Dr. Clifford Jack, Jr.
Clifford Jack, Jr., M.D.
"What's exciting now is that we're looking even earlier — before symptoms begin — to see if we can predict who might be at greatest risk of developing cognitive problems in the future," says Clifford Jack, Jr., M.D., radiologist and lead author of the study.

The new prediction model combined several factors, including age, sex, genetic risk as associated with APOE genotype and brain amyloid levels detected on PET scans. Using the data, researchers can calculate an individual's likelihood of developing MCI or dementia within 10 years or over the predicted lifetime. Of all the predictors evaluated, the brain amyloid levels detected on PET scans was the predictor with the largest effect for lifetime risk of both MCI and dementia.

## Data Analysis
Investigation into the dataset revealed 89 persons with "Mild Cognitive Imparment" where 47 progress to alzhiemer's and 42 do not. 