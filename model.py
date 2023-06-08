import pandas as pd
import streamlit as st
import pickle
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score


X = pd.read_csv('./final_task/data/titanic.csv')
y = X['Target']
X.drop(columns=['Target'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state = 42)

X_train.info()

cat_columns = []
num_columns = []

for column_name in X_train.columns:
    if X_train[column_name].dtypes == object:
        cat_columns += [column_name]
    else:
        num_columns += [column_name]


cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder())])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_columns),
        ('cat', cat_transformer, cat_columns)])

preprocessor.fit(X_train, y_train)

test = preprocessor.transform(X_test)
train = preprocessor.transform(X_train)

LR = LogisticRegression(
    fit_intercept=True, random_state=42, solver='liblinear')
LR.fit(train, y_train)

with open("model.pkl", 'wb') as file:
    pickle.dump(LR, file)

with open('model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

scores_train = cross_validate(pickle_model, train, y_train, scoring='f1',
                        cv=ShuffleSplit(n_splits=5, random_state=42))

score_test = f1_score(y_test, pickle_model.predict(test))

st.title('Результаты работы модели')
result = st.button('Рассчитать Score')
if result:
    DF_cv_linreg = pd.DataFrame(scores_train)
    st.write('Результаты кросс-валидации', '\n', DF_cv_linreg, '\n')
    st.write('Среднее f1 на кросс-валидации -', round(DF_cv_linreg.mean()[2], 2))
    st.write('f1 на тестовых данных - ', round(score_test, 2))
