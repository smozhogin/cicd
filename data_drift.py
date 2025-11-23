import datetime
import pandas as pd
from pathlib import Path
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

BASE_DIR = Path('APP')
REPORTS_DIR = BASE_DIR/'reports'
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

EVIDENTLY_REPORT_PATH = REPORTS_DIR/'evidently_iris_data_drift.html'

def load_iris_data_with_drift():
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Загружаем датасет Iris
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    df = X.copy()
    df['target'] = y

    # reference_data, current_data = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['target'])

    # Изменяем один из признаков в current
    modified_data = df.copy()
    modified_data['sepal length (cm)'] = modified_data['sepal length (cm)'] * 1.2

    return df, modified_data

def main():
    df, modified_data = load_iris_data_with_drift()

    data_report = Report(
        metrics=[
            DataDriftPreset(stattest='psi', stattest_threshold=0.3),
            DataQualityPreset(),
        ],
        timestamp=datetime.datetime.now(),
    )

    data_report.run(
        reference_data=df,
        current_data=modified_data,
    )

    display(data_report)

    data_report.save_html(str(EVIDENTLY_REPORT_PATH))

if __name__ == "__main__":
    main()