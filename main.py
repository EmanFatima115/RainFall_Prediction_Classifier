

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.core.magics import display
from pandas.core.groupby import categorical
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score)

df=pd.read_csv('weather AUS.csv')
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df['RainTomorrow'].value_counts())
df=(df.dropna(subset=['RainTomorrow']))
df['RainTomorrow']=df['RainTomorrow'].map({
    'No':0,
    'Yes':1
})
df['Date']=pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year
df['month']=df['Date'].dt.month
df['Day']=df['Date'].dt.day
df=df.drop(columns=['Date'])
x=df.drop(columns=['RainTomorrow'])
y=df['RainTomorrow']
numeric_featues=x.select_dtypes(include=['int64','float64']).columns.to_list()
categorical_features=x.select_dtypes(include=['object','str']).columns.to_list()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
numeric_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])
preprocessor=ColumnTransformer(transformers=[
    ('num',numeric_transformer,numeric_featues),
    ('cat',categorical_transformer,categorical_features)
])
logistic_Pipeline=Pipeline(steps=[
    ('preprocesssor',preprocessor),
    ('classifier',LogisticRegression(
        max_iter=1000,
    ))
])
logistic_Pipeline.fit(x_train,y_train)
print('Logistic Regression Training Completed:')
RandomForest_Pipeline=Pipeline(steps=[
    ('preprocesssor',preprocessor),
    ('classifier',RandomForestClassifier(
        random_state=42,
    ))
])
param_grid={
    'classifier__n_estimators':[50,100],
    'classifier__max_depth':[None,3],
    'classifier__min_samples_split':[2,4]
}
grid_search=GridSearchCV(
    estimator=RandomForest_Pipeline,
    param_grid=param_grid,
    cv=3,scoring='f1',n_jobs=1
)
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
logistic_cv_scores=cross_val_score(
    logistic_Pipeline,
    x_train,y_train,cv=3,scoring='f1'
)
logistic_cv_f1=logistic_cv_scores.mean()
random_forest_cv_f1=grid_search.best_score_
if logistic_cv_f1 >= random_forest_cv_f1:
    best_model = logistic_Pipeline
    model_name = "Logistic Regression"
else:
    best_model = RandomForest_Pipeline
    model_name = "Random Forest"
y_pred=best_model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print(classification_report(y_test,y_pred,target_names=['NoRain','Rain']))

cm=confusion_matrix(y_test,y_pred)
display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['NoRain','Rain'])
display.plot(cmap='Blues')
plt.title('RainFall Prediction Matrix:')
plt.show()
sns.countplot(data=df, x="RainTomorrow")
plt.title("Rain Tomorrow Distribution")
plt.xlabel("Rain Tomorrow (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.show()
y_probability = best_model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probability)
auc_score=roc_auc_score(y_test,y_probability)
joblib.dump(best_model,'rainfall_model.pkl')
loaded_model=joblib.load('rainfall_model.pkl')
new_weather = pd.DataFrame([{
    "Location": "Sydney",
    "MinTemp": 18.0,
    "MaxTemp": 26.0,
    "Rainfall": 0.0,
    "Evaporation": 5.0,
    "Sunshine": 8.0,
    "WindGustDir": "North",
    "Year": 2020,
    "Month": 1,
    "Day": 21
}])
prediction=best_model.predict(new_weather)
if prediction[0] == 1:
    print("\nPrediction: It may rain tomorrow.")
else:
    print("\nPrediction: It may not rain tomorrow.")