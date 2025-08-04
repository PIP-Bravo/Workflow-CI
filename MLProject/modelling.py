import argparse
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ambil parameter dari command line
parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations for logistic regression")
parser.add_argument("--data_dir", type=str, default="mushrooms_preprocessing", help="Path to the preprocessed dataset folder")
args = parser.parse_args()

# Muat data dari direktori yang ditentukan
X_train = pd.read_csv(f"{args.data_dir}/X_train.csv")
X_test = pd.read_csv(f"{args.data_dir}/X_test.csv")
y_train = pd.read_csv(f"{args.data_dir}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{args.data_dir}/y_test.csv").values.ravel()

# Contoh input untuk dokumentasi model
input_example = X_train.head(5)

# Mulai MLflow Run
with mlflow.start_run():
    mlflow.autolog()

    # Inisialisasi dan training model
    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # Logging model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
