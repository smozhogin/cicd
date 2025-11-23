import pandas as pd
from pathlib import Path
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import train_test_validation

BASE_DIR = Path('APP')
REPORTS_DIR = BASE_DIR/'reports'
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

DEEPCHECKS_REPORT_PATH = REPORTS_DIR/'deepchecks_iris_train_test.html'

def load_iris_train_test():
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Загружаем датасет Iris
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    train_df = X_train.copy()
    train_df['target'] = y_train

    test_df = X_test.copy()
    test_df['target'] = y_test

    return train_df, test_df

def main():
    train_df, test_df = load_iris_train_test()

    train_ds = Dataset(train_df, label='target', cat_features=[])
    test_ds = Dataset(test_df, label='target', cat_features=[])

    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)

    display(result)

    result.save_as_html(str(DEEPCHECKS_REPORT_PATH))

if __name__ == "__main__":
    main()