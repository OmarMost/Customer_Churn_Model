import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

# Set plot style
sns.set(style="whitegrid")

# Load data
file_path = 'customer_churn_dataset.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Handling Missing Values
df['internet_service'].fillna('No', inplace=True)
df['tech_support'].fillna('No', inplace=True)
df['monthly_data_gb'].fillna(0, inplace=True)

# Feature Engineering / Selection
X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "params": {}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=sum(y==0)/sum(y==1)),
        "params": {
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5],
            'classifier__n_estimators': [50, 100]
        }
    }
}

results = {}

for name, mp in models_params.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', mp['model'])])
    
    if mp['params']:
        print(f"Tuning {name}...")
        clf = RandomizedSearchCV(pipeline, mp['params'], n_iter=5, cv=3, scoring='f1', random_state=42, n_jobs=-1)
    else:
        clf = pipeline
    
    clf.fit(X_train, y_train)
    
    if hasattr(clf, 'best_estimator_'):
        best_model = clf.best_estimator_
        print(f"Best params for {name}: {clf.best_params_}")
    else:
        best_model = clf
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {"F1-Score": f1, "ROC-AUC": roc_auc}
    
    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")

results_df = pd.DataFrame(results).T[['F1-Score', 'ROC-AUC']]
print("\nModel Comparison:")
print(results_df)

best_model_name = results_df['F1-Score'].idxmax()
print(f"\nBest Model based on F1-Score: {best_model_name}")
