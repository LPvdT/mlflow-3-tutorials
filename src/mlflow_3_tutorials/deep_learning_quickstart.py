import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main() -> None: ...


# Load Iris dataset and prepare the DataFrame
iris = load_iris()
iris_df, iris_target = (
    pd.DataFrame(iris.data, columns=iris.feature_names),  # type: ignore
    iris.target,  # type: ignore
)

# Split the dataset into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    iris_df, iris_target, test_size=0.2, random_state=42
)
