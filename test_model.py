from sklearn.metrics import f1_score
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from model import preprocessing, model_training


@pytest.fixture(params=['./data/titanic.csv'])
def load_data(request):
    df = pd.read_csv(request.param)
    return df


def test_model(load_data):
    y = load_data['Target']
    X = load_data.loc[:, load_data.columns != 'Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    preprocessor = preprocessing(X_train, y_train)
    test = preprocessor.transform(X_test)
    train = preprocessor.transform(X_train)

    LR = model_training(train, y_train)

    score_test = f1_score(y_test, LR.predict(test))
    assert score_test >= 0.6
    assert score_test <= 1.0
