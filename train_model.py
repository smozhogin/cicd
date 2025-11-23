# Обучаем модель на датасете Iris
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data():
    # Загружаем датасет Iris
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    return X, y

def train_model(X, y, params):
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Создаем модель XGBoost для классификации
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        **params
    )

    # Обучаем модель
    model.fit(X_train, y_train)

    # Делаем предсказания
    y_pred = model.predict(X_test)

    # Оцениваем метрики
    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }

    report = classification_report(y_test, y_pred)

    return model, metrics, report

def main():
    BASE_DIR = Path('APP')
    BASE_DIR.mkdir(exist_ok=True)

    MLFLOW_DIR = BASE_DIR/'mlruns'

    os.environ['MLFLOW_TRACKING_URI'] = f'file:{MLFLOW_DIR}'
    mlflow.set_experiment('iris_xgb_gridsearch')

    # Задаем сетку параметров для перебора
    param_grid = [
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.9},
        {'n_estimators': 150, 'learning_rate': 0.2, 'max_depth': 3, 'subsample': 0.9, 'colsample_bytree': 0.8},
        {'n_estimators': 250, 'learning_rate': 0.15, 'max_depth': 4, 'subsample': 0.85, 'colsample_bytree': 0.9},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.9, 'colsample_bytree': 0.7},
        {'n_estimators': 120, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 180, 'learning_rate': 0.2, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'n_estimators': 220, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.9, 'colsample_bytree': 0.8},
        {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.9},
    ]

    X, y = load_data()

    best_score = -1
    best_run_id = None
    best_params = None

    for idx, params in enumerate(param_grid, start=1):
        print(f'\nЗапуск эксперимента {idx}/10 с параметрами: {params}')

        with mlflow.start_run(run_name=f'run_{idx}') as run:
            model, metrics, report = train_model(X, y, params)

            # Логируем параметры
            mlflow.log_params(params)

            # Логируем метрики
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Логируем модель
            mlflow.sklearn.log_model(model, artifact_path='model')

            # Логируем отчет
            reports_dir = BASE_DIR /'reports'
            reports_dir.mkdir(exist_ok=True, parents=True)

            report_path = reports_dir / f'classification_report_run_{idx}.txt'
            report_path.write_text(report)

            mlflow.log_artifact(str(report_path), artifact_path='reports')

            # Отслеживаем лучшую модель
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_params = params
                best_run_id = run.info.run_id

    print('\nЛучший результат:')
    print('Run ID:', best_run_id)
    print('Accuracy:', best_score)
    print('Параметры:', best_params)

if __name__ == "__main__":
    main()