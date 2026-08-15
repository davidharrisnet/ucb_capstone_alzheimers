# Note this is is active development, and will stabalize by 9 AM August 15th.

# UCB Capstone Project 
U.C. Berkeley Engineering 
## Protien predictors for developing Alzheimers


### Abstract 
The potential of developing of Alzheimer's disease can be predicted by two key proteins in the brain - amyloid and tau. 
[[Mayo Clinic]](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/)
This report uses the Kaggle dataset [Plasma lipidomics in Alzheimer's disease](https://www.kaggle.com/datasets/fereshtehjozaghkar/plasma-lipidomics-in-alzheimers-disease), which contains critical Tau and Amyloid measures of 212 real patient data to predict 'Progression to Alzheimer's Disease'.

The detailed analysis for this project is contained in the jupyter notebook [Jupyter Notebook  Lipidomics](lipidomics_notebook.ipynb).
This report folows the steps of the CRISP-DM methodology, and contains the following sections:


1. Business Understanding
3. Data Understanding
4. Data Preparation
5. Modeling

    1. Validating the most predicitive parameters of "Progression to Alzheimer's"
    2. Determine the best classification algorithm
    3. Investigate Hyper Parameters 
    4. Minimizing False Negatives
6. Evaluation
8. Conclusion
9. References


# Introduction


# Business Understanding
### Clinical Background

[Mayo Clinic](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-scientists-create-tool-to-predict-alzheimers-risk-years-before-symptoms-begin/)

Predicting Alzheimer's disease
Alzheimer's disease is marked by two key proteins in the brain: amyloid, which forms plaques, and tau, which forms tangles. Drugs recently approved by the Food and Drug Administration remove amyloid from the brain and can slow the rate of disease progression for people with MCI or mild dementia.

Photo of Dr. Clifford Jack, Jr.
Clifford Jack, Jr., M.D.
"What's exciting now is that we're looking even earlier — before symptoms begin — to see if we can predict who might be at greatest risk of developing cognitive problems in the future," says Clifford Jack, Jr., M.D., radiologist and lead author of the study.

The new prediction model combined several factors, including age, sex, genetic risk as associated with APOE genotype and brain amyloid levels detected on PET scans. Using the data, researchers can calculate an individual's likelihood of developing MCI or dementia within 10 years or over the predicted lifetime. Of all the predictors evaluated, the brain amyloid levels detected on PET scans was the predictor with the largest effect for lifetime risk of both MCI and dementia.

# Data Understanding

* Image of "Diagnostic" bar chart.
* Image of "Mild Cognitive Impariment" bar chart



Investigation into the dataset revle"Mild Cognitive Imparment" where 47 progress to alzhiemer's and 42 do not. 

Concerns this is a small dataset

It is relatively well balanced, but given the small size its still something to consider. 

Perhaps you should preemptively balance the set of Yes/No, but randomly selecting 42 Yeses?

* Image of the scatter plot of Progress to Alzheimers for CSF Amalyoid and CSF Phosphorylated tau.
 Using Logistic Regression, evaluate the power of the pvalue of each parameter
```
Feature(s)   AUC  p-value Significant (p<0.05)
                Age 0.550   0.3656                   No
                Sex 0.524   0.5195                   No
               MMSE 0.640   0.0360                  Yes
              APOE4 0.727   0.0010                  Yes
        CSF Amyloid 0.782   0.0010                  Yes
      CSF Total tau 0.728   0.0010                  Yes
Amyloid + Total tau 0.810   0.0010                  Yes
```
# Data Preparation

Some "Mild Cognitive Impairment" had Nan values which were filled with the mode value.

# Modeling

1. Using Logistical Regression, evaluate MMSE, APOE, and combinations of Amyloid and Tau to see which are the most predicative of Progression to Alzheiemrs. 

* MMSE
* APOE4

* CSF Amyloid
```

             Pred: No  Pred: Yes
Actual: No         25         17
Actual: Yes         8         39
                precision    recall  f1-score   support

No progression      0.758     0.595     0.667        42
    Progressed      0.696     0.830     0.757        47

      accuracy                          0.719        89
    

Accuracy: 0.7191011235955056
AUC: 0.7510131712259374

```

* CSF Total tau
```

             Pred: No  Pred: Yes
Actual: No         33          9
Actual: Yes        16         31
                precision    recall  f1-score   support

No progression      0.673     0.786     0.725        42
    Progressed      0.775     0.660     0.713        47

      accuracy                          0.719        89
    
Accuracy: 0.7191011235955056
AUC: 0.7272036474164133

```
* CSF PPhosphorylated tau
```
Actual: No         23         19
Actual: Yes        13         34
                precision    recall  f1-score   support

No progression      0.639     0.548     0.590        42
    Progressed      0.642     0.723     0.680        47

      accuracy                          0.640        89
    
Accuracy: 0.6404494382022472
AUC: 0.7155521783181358

```

* Tau and Total Tau
```
Actual: No         33          9
Actual: Yes        16         31
                precision    recall  f1-score   support

No progression      0.673     0.786     0.725        42
    Progressed      0.775     0.660     0.713        47

      accuracy                          0.719        89
   

Accuracy: 0.7191011235955056
AUC: 0.7206180344478218


```

* Amyloid and Total Tau
```
      Pred: No  Pred: Yes
Actual: No         29         13
Actual: Yes        11         36
                precision    recall  f1-score   support

No progression      0.725     0.690     0.707        42
    Progressed      0.735     0.766     0.750        47

      accuracy                          0.730        89
   
Accuracy: 0.7303370786516854
AUC: 0.8011651469098278

```
### Evaluation

* CSF Phosphorylated tau (pg/mL) is just above random, CSF Total tau (pg/mL) alone is better
* Amyloid and Total Tau are the strongest Accuracy and AUC. 
* A little concerned about the false negative rate of 16
* Is the higher Yes count inflating the accuracy? 

## Classification

Given the choice to use Total tau, and Amyloid, what is the best classification algorithm.
### Classification with default parameters.
The classification algorithms with default parameters are evaluteed
      

  | Model | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| LDA | 0.802 | 0.730 | 0.735 | 0.766 |
| Logistic Regression | 0.801 | 0.730 | 0.735 | 0.766 |
| SVM (RBF) | 0.801 | 0.730 | 0.780 | 0.681 |
| Naive Bayes | 0.799 | 0.742 | 0.750 | 0.766 |
| Gradient Boosting | 0.799 | 0.775 | 0.814 | 0.745 |
| Random Forest | 0.772 | 0.730 | 0.780 | 0.681 |
| K-Nearest Neighbors | 0.754 | 0.708 | 0.723 | 0.723 |     

### Classificaiton with hyper parameters
Each classificaion algorithm was investigated and hyper parameters were adjusted
**Explain each one and why they wer applied.**

 Logistic Regression
 ```
 Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000,C=0.01))]),
 ```
Random Forest    
```
    RandomForestClassifier(n_estimators=300, random_state=42, max_depth=2), # max_depth=2
```
SVM (RBF) 
```           
Pipeline([('sc', StandardScaler()), ('clf', SVC(C=0.1, gamma=0.01, probability=True, random_state=42))]),
```
Gradient Boosting

```
GradientBoostingClassifier(random_state=42,max_depth=2, learning_rate=0.05, n_estimators=200),
```

K-Nearest Neighbors 
```
Pipeline([('sc', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=7))]), #n_neighbors=7
```


LDA

```
Pipeline([('sc', StandardScaler()), ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))]),
```
#### Results
Remarkably, Gradient Boosting, and Random Forest now outperfor Logistic Regression with hyper parameters. This is not what I expected!

| Model | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Gradient Boosting | 0.825 | 0.742 | 0.786 | 0.702 |
| Random Forest | 0.810 | 0.753 | 0.778 | 0.745 |
| Logistic Regression | 0.805 | 0.663 | 0.631 | 0.872 |
| LDA | 0.802 | 0.730 | 0.735 | 0.766 |
| Naive Bayes | 0.799 | 0.742 | 0.750 | 0.766 |
| SVM (RBF) | 0.794 | 0.528 | 0.528 | 1.000 |
| KNN | 0.774 | 0.716 | 0.729 | 0.745 |

## Evaluation
A detailed analysis of the results show some things to consider.
* Logistic Regression with hyper parameters did worse. ( look at this again. )
* Gradient Boosting and Random Forest have higher false negatives
* 

## False Negatives
In medical analysis false negtives are really bad. We are predicting someone will not progress to alzhiemers and therefore not be treated.  This brings up a lot of ethical considerations. How expensive is preventative treatment for alzhiemers? Could we just treat everyone? At any rate lets look at one means of affecting the false negative rate. (one that I understand)
### Logistic Regression thresholds.

