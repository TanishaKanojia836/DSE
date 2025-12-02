#Use Naive bayes, K-nearest, and Decision tree classification algorithms to build classifiers
on any two datasets. Pre-process the datasets using techniques specified in Q2. Compare the
Accuracy, Precision, Recall and F1 measure reported for each dataset using the
abovementioned classifiers under the following situations:
i. Using Holdout method (Random sampling):
a) Training set = 80% Test set = 20%
b) Training set = 66.6% (2/3rd of total), Test set = 33.3%
ii. Using Cross-Validation:
a) 10-fold
b) 5-fold
# CLASSIFICATION ON IRIS & WINE DATASETS

!pip install scikit-learn --quiet
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# LOAD IRIS & WINE DATASETS

iris = load_iris()
wine = load_wine()

def to_dataframe(ds):
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target)
    return X, y

X_iris, y_iris = to_dataframe(iris)
X_wine, y_wine = to_dataframe(wine)

# PRE-PROCESSING
scaler = StandardScaler()

def preprocess(X):
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

X_iris = preprocess(X_iris)
X_wine = preprocess(X_wine)

# Models

models = {
    "NaiveBayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

# EVALUATION FUNCTION
def evaluate(X, y, dataset_name):
    print("\n=====================================")
    print(f"RESULTS FOR {dataset_name}")
    print("=====================================\n")

    #HOLDOUT 80/20
    print("----- HOLDOUT: 80% Train / 20% Test -----")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print(f"\n{name}:")
        print("Accuracy :", accuracy_score(y_test, pred))
        print("Precision:", precision_score(y_test, pred, average='weighted'))
        print("Recall   :", recall_score(y_test, pred, average='weighted'))
        print("F1 Score :", f1_score(y_test, pred, average='weighted'))

    # HOLDOUT 66.6 / 33.3
    print("\n----- HOLDOUT: 66.6% Train / 33.3% Test -----")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print(f"\n{name}:")
        print("Accuracy :", accuracy_score(y_test, pred))
        print("Precision:", precision_score(y_test, pred, average='weighted'))
        print("Recall   :", recall_score(y_test, pred, average='weighted'))
        print("F1 Score :", f1_score(y_test, pred, average='weighted'))

    # CROSS VALIDATION
    print("\n----- CROSS VALIDATION -----")
    for folds in [10, 5]:
        print(f"\n--- {folds}-Fold CV ---")
        for name, model in models.items():
            scores = cross_validate(
                model, X, y, cv=folds,
                scoring=('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted')
            )

            print(f"\n{name}:")
            print("Accuracy :", scores['test_accuracy'].mean())
            print("Precision:", scores['test_precision_weighted'].mean())
            print("Recall   :", scores['test_recall_weighted'].mean())
            print("F1 Score :", scores['test_f1_weighted'].mean())

# RUN FOR IRIS & WINE
evaluate(X_iris, y_iris, "IRIS DATASET")
evaluate(X_wine, y_wine, "WINE DATASET")
