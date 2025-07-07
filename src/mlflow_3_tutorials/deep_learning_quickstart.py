from typing import cast

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from mlflow.data import pandas_dataset
from mlflow.entities import Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlflow_3_tutorials.lib.dl_utils import IrisClassifier, prepare_data


def main() -> None: ...


# Load Iris dataset and prepare the DataFrame
iris = load_iris()
iris_df, iris_target = (
    pd.DataFrame(iris.data, columns=iris.feature_names),  # type: ignore
    iris.target,  # type: ignore
)

# Split into training and testing datasets
train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)
train_df = cast("pd.DataFrame", train_df)
test_df = cast("pd.DataFrame", test_df)

# Prepare training data
train_dataset = pandas_dataset.from_pandas(train_df, name="train")
X_train, y_train = prepare_data(train_dataset.df)

# Define the PyTorch model and move it to the device
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(iris.target_name)  # type: ignore
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scripted_model = IrisClassifier(input_size, hidden_size, output_size).to(device)
scripted_model = torch.jit.script(scripted_model)
