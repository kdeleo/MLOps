import pickle

from pathlib import Path

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import mlflow

url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment") 
mlflow.sklearn.autolog()

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

# Load the data
df = pd.read_parquet(url)

# Number of rows
num_records = len(df)
print(f"Number of records: {num_records}")

df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

df = df[(df.duration >= 1) & (df.duration <= 60)]

categorical = ['PULocationID', 'DOLocationID']
df[categorical] = df[categorical].astype(str)

num_records_train = len(df)
print(f"Number of records for training: {num_records_train}")

dicts = df[categorical ].to_dict(orient='records')

dv = DictVectorizer()
X = dv.fit_transform(dicts)

target = 'duration'
y = df[target].values

with mlflow.start_run() as run:
    lr = LinearRegression()
    lr.fit(X, y)

    dv_path = models_folder / "dv.pkl"
    with open(dv_path, "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact(dv_path, artifact_path="preprocessor")

    mlflow.sklearn.log_model(lr, artifact_path="linear_model")

    print(f"MLflow run_id: {run.info.run_id}")
