from feature_selector import load_and_clean_data, run_select_k_best

from model_trainer import train_model

if __name__ == "__main__":
    X, y = load_and_clean_data("C:/Users/LENOVO/OneDrive/Documents/shubham-personal/OneDrive/Desktop/git_projects/PulseAI/diabetes.csv")
    top_features = run_select_k_best(X, y)
    # print("\nTop selected features:", top_features)
    train_model(X,y, top_features)
