import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_clean_data(path="C:/Users/LENOVO/OneDrive/Documents/shubham-personal/OneDrive/Desktop/git_projects/PulseAI/diabetes.csv"):
    df = pd.read_csv(path)

    # Replace 0s in key columns with NaN
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

    # Fill missing values with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df=df.convert_dtypes()

    X = df.drop("Outcome", axis=1).astype(float)
    y = df["Outcome"].astype(int)
    return X, y

def run_select_k_best(X, y, k=5, save_plot=True):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    scores = selector.scores_

    # Show scores
    print("\n📊 Top features by SelectKBest:")
    for feature, score in zip(X.columns, scores):
        print(f"{feature}: {score:.2f}")

    if save_plot:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=X.columns, y=scores)
        plt.title("Feature Importance (SelectKBest F-Scores)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("selectkbest_scores.png")
        print("✅ Plot saved: selectkbest_scores.png")
        # plt.show()
    return selected_features.tolist()

