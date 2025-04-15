# Dry Beans  classification
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Objective](#objective)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [How to Run](#how-to-run)

## Introduction
![alt text](image.png)
- Seven different types of dry beans were used in this research, taking into account the features such as form, shape, type, and structure by the market situation. A computer vision system was developed to distinguish seven different registered varieties of dry beans with similar features in order to obtain uniform seed classification. For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. Bean images obtained by computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.
## Dataset
[dataset](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset/data)
## objective
- The objective of this project is to develop a machine learning model that can accurately classify seven different types of dry beans based on their visual characteristics such as shape and size. 
- This classification will help in automating the sorting process in the agriculture industry, ensuring better quality control and reducing manual labor.
## Project Workflow
1. *Data Loading*: The dataset is loaded into a pandas DataFrame.

2. *Exploratory Data Analysis (EDA)*:
   - Display sample rows, data dimensions, and basic info.
   - Show the unique classes of beans.
   - Count missing and duplicated values.
   - Remove any duplicate entries and update the dataset size.

3. *Data Visualization*:
   - Plot the distribution of bean classes using a histogram.
   - Plot a heatmap to visualize feature correlations.
   - Visualize boxplots for feature value distributions.

4. *Label Encoding*:
   - Convert the categorical target variable (Class) into numeric labels using LabelEncoder.

5. *Feature Scaling*:
   - Standardize the feature values using StandardScaler to ensure consistent scale.

6. *Dimensionality Reduction (PCA)*:
   - Apply PCA to identify the number of principal components needed to preserve 95% of the data variance.
   - Transform the feature space accordingly and create a new DataFrame containing the selected principal components and the target variable.

7. *Preprocessed Data Retrieval*:
   - The final processed dataset is stored internally and can be accessed using the get_data() method for model training.
8. *Train-Test Split*:
   - Split the dataset into training and testing sets using an 80/20 ratio.

9. *Model Training and Hyperparameter Tuning*:
   - *Random Forest*: Trained using RandomForestClassifier with GridSearchCV to tune hyperparameters like n_estimators, criterion, and max_depth.
   - *Support Vector Machine (SVM)*: Trained using SVC with GridSearchCV to tune kernel and C parameters.

10. *Model Evaluation*:
    - Evaluate the best model (selected via grid search) using the test set.
    - Display confusion matrix and classification report (precision, recall, F1-score for each class).

11. *Model Selection*:
    - The best performing model (based on F1-macro score) is selected and can be used for final predictions or deployment.
## Results
- Two machine learning models were trained and evaluated on the dataset after preprocessing and dimensionality reduction using PCA:

1. *Support Vector Classifier (SVC)*

        Best Parameters: C=10, class_weight='balanced'

        Accuracy: 89%

        Macro Average F1-score: 0.89

        Weighted Average F1-score: 0.89


2. *Random Forest Classifier*

        Best Parameters: n_estimators=200, max_depth=10, criterion='entropy', class_weight='balanced'

        Accuracy: 89%

        Macro Average F1-score: 0.89

        Weighted Average F1-score: 0.89


- ## Conclusion

 - Both models achieved the same accuracy and macro/weighted F1-score.

 - Random Forest slightly outperformed SVC in precision/recall for some classes, especially in class 3 and 5.

 - Depending on the deployment need (speed vs interpretability vs memory usage), either model can be considered.
## How to Run
1. Creation of virtual environments
```Bash
python -m venv <name of your environment>
```
2. activation of environment
```Bash
<name of your environment>\Scripts\activate
```
3. Change the directory inside to the environment
```Bash 
cd <name of your environment>
```
4. creat folder in this directory
```Bash 
md src
```
5. Change the directory inside to src
```Bash
cd src
```
6. Colne this repository:
```Bash
git clone <url of repo >
```
7. install the required dependencies:
```Bash 
pip install -r requirements.txt
```
8. Run the credit_card_fraud_detection.py script:
```Bash 
python main.py <path of dataset>
```