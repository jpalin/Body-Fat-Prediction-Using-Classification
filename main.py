import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# -------------------------------
# Data Extraction & Preprocessing
# -------------------------------

def import_dataset():
    """
    Imports the body fat dataset from CSV and drops irrelevant columns.
    Filters out rows where age is 60 or older to focus on the target demographic.
    """
    body_data = pd.read_csv("data/bodyfat.csv")
    body_data = body_data.drop(["Density"], axis=1)
    body_data = body_data[body_data["Age"] < 60]
    return body_data


def remove_incorrect_data(body_data):
    """
    Cleans the dataset by removing rows with implausible body fat or height values.
    This ensures model training is based on realistic data points.
    """
    body_data = body_data.loc[body_data['BodyFat'] != 0.0]
    body_data = body_data.loc[body_data['Height'] != 29.50]
    return body_data


def add_bmi_column(body_data):
    """
    Calculates BMI (Body Mass Index) using the formula: (weight / height^2) * 703
    Adds the result as a new column to the dataset.
    """
    body_data["BMI"] = (body_data["Weight"] / body_data["Height"]**2) * 703
    return body_data


def convert_waist(body_data):
    """
    Converts the 'Abdomen' measurement from centimeters to inches.
    Stores the result in a new column named 'waist'.
    """
    body_data["waist"] = body_data["Abdomen"] / 2.54
    return body_data


def calc_height_to_waist(body_data):
    """
    Calculates the height-to-waist ratio, which is a useful indicator of body fat percentage.
    The result is rounded to 2 decimal places for simplicity.
    """
    body_data["heightToWaistRatio"] = round(body_data["waist"] / body_data["Height"], 2)
    return body_data


def replace_body_fat_col(body_data):
    """
    Converts the continuous 'BodyFat' variable into a binary classification target.
    Labels individuals as 'yes' if BodyFat < 22% (considered healthy), otherwise 'no'.
    """
    body_data["HealthyBodyFat?"] = np.where(body_data["BodyFat"] < 22, 'yes', 'no')
    return body_data

# -------------------------------
# Feature Selection & Preparation
# -------------------------------

def select_cols(body_data, columns):
    """
    Selects and retains only the specified columns in the dataset.
    Args:
        body_data (DataFrame): The original dataset.
        columns (list): List of column names to retain.
    """
    return body_data[columns]


def define_vars(body_data, target_col):
    """
    Splits the dataset into features (X) and target (y) arrays.
    Args:
        body_data (DataFrame): The preprocessed dataset.
        target_col (str): Name of the column to be used as the target variable.
    Returns:
        Tuple[np.array, np.array]: X (features), y (target)
    """
    y = body_data.loc[:, target_col].values
    x = body_data.drop(columns=[target_col]).values
    return y, x

# -------------------------------
# Model Training & Evaluation
# -------------------------------

def training_algorithms(y, x):
    """
    Splits the data into training and testing sets and fits multiple classifiers:
    Logistic Regression, Random Forest, and K-Nearest Neighbors.
    Returns all trained models along with test data.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    clf1 = LogisticRegression()
    clf1.fit(x_train, y_train)

    clf2 = RandomForestClassifier()
    clf2.fit(x_train, y_train)

    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf3.fit(x_train, y_train)

    return clf1, clf2, clf3, x_test, y_test


def evaluate_algorithms(clf1, clf2, clf3, x_test, y_test):
    """
    Evaluates the performance of each classifier on the test set and returns the best model
    based on highest accuracy.
    """
    classifiers = [clf1, clf2, clf3]
    names = ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"]
    scores = {}

    for name, clf in zip(names, classifiers):
        score = clf.score(x_test, y_test)
        scores[name] = score
        print(f"{name}: {score:.4f}")

    best_model_name = max(scores, key=scores.get)
    print(f"\nBest Performing Model: {best_model_name} with Accuracy = {scores[best_model_name]:.4f}")
    
    return classifiers[names.index(best_model_name)]

# -------------------------------
# Pipeline Orchestration
# -------------------------------

def main():
    """
    Main function to run the entire data processing and model training pipeline.
    Returns the best trained model based on accuracy.
    """
    # Data loading and preprocessing
    body_data = import_dataset()
    body_data = remove_incorrect_data(body_data)
    body_data = add_bmi_column(body_data)
    body_data = convert_waist(body_data)
    body_data = calc_height_to_waist(body_data)
    body_data = replace_body_fat_col(body_data)

    # Feature selection (after some analysis)
    body_data = select_cols(body_data, ["waist", "Wrist", "heightToWaistRatio", "HealthyBodyFat?"])

    # Define features and target
    y, x = define_vars(body_data, "HealthyBodyFat?")

    # Train and evaluate models
    clf1, clf2, clf3, x_test, y_test = training_algorithms(y, x)
    best_model = evaluate_algorithms(clf1, clf2, clf3, x_test, y_test)

    return best_model

# -------------------------------
# Script Entry Point
# -------------------------------

if __name__ == "__main__":
    # Run pipeline and get best model
    best_model = main()

    # Serialize model to disk for future use
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Load model and test prediction (example)
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    prediction = loaded_model.predict([[30, 16, 47]])  # Example input: [waist, Wrist, heightToWaistRatio]
    print("\nPrediction for input [30, 16, 47]:", prediction[0])
