import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

print("="*65)
print("  SPPS — Retraining on Real Dataset (10,000 Students)")
print("="*65)

df = pd.read_csv('data/student_records.csv')
print(f"\n[1] Loaded: {df.shape[0]} records | {df['Department'].nunique()} departments")

# Derive composite performance score
df['Composite'] = (
    (df['CA_Score'] / 30 * 40) +
    (df['MidSem_Score'] / 20 * 20) +
    (df['Prev_CGPA'] / 5.0 * 25) +
    (df['Attendance_Rate'] / 100 * 10) +
    (df['Study_Hours_Week'] / 40 * 5)
).clip(0, 100)

def classify(s):
    if s >= 70: return 'Distinction'
    elif s >= 60: return 'Credit'
    elif s >= 45: return 'Pass'
    else: return 'At-Risk'

df['Performance_Class'] = df['Composite'].apply(classify)
print(f"\n[2] Class distribution:")
print(df['Performance_Class'].value_counts().to_string())

df_model = df.drop(['Student_ID','Student_Name'], axis=1)
le_gender = LabelEncoder()
df_model['Gender'] = le_gender.fit_transform(df_model['Gender'])
df_model = pd.get_dummies(df_model, columns=['Department'], drop_first=False)

X = df_model.drop(['Performance_Class','Composite'], axis=1)
y = df_model['Performance_Class']
feature_names = list(X.columns)
dept_columns  = [c for c in feature_names if c.startswith('Department_')]
departments   = sorted(df['Department'].unique().tolist())

X_np  = X.to_numpy()
imp   = SimpleImputer(strategy='median')
X_imp = imp.fit_transform(X_np)
sc    = StandardScaler()
X_sc  = sc.fit_transform(X_imp)

smote = SMOTE(random_state=42)
X_b, y_b = smote.fit_resample(X_sc, y)
print(f"\n[3] After SMOTE: {X_b.shape[0]} records")

X_tr,X_te,y_tr,y_te = train_test_split(X_b,y_b,test_size=0.2,random_state=42,stratify=y_b)

models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=2, max_features='sqrt', random_state=42),
}

results = {}
print("\n[4] Training all models...")
for name, m in models.items():
    m.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, m.predict(X_te))
    rep = classification_report(y_te, m.predict(X_te), output_dict=True)
    results[name] = {
        'accuracy':  round(acc*100,2),
        'precision': round(rep['weighted avg']['precision']*100,2),
        'recall':    round(rep['weighted avg']['recall']*100,2),
        'f1':        round(rep['weighted avg']['f1-score']*100,2),
    }
    print(f"    {name:<25} -> {acc*100:.2f}%")

best = models['Random Forest']
print(f"\n[5] Random Forest Report:")
print(classification_report(y_te, best.predict(X_te)))

feat_imp = sorted(zip(feature_names, best.feature_importances_), key=lambda x:-x[1])

joblib.dump(best,      'models/rf_model.pkl')
joblib.dump(sc,        'models/scaler.pkl')
joblib.dump(imp,       'models/imputer.pkl')
joblib.dump(le_gender, 'models/le_gender.pkl')

metadata = {
    'feature_names':       feature_names,
    'dept_columns':        dept_columns,
    'departments':         departments,
    'classes':             list(best.classes_),
    'model_results':       results,
    'feature_importances': [(f, round(float(i)*100,2)) for f,i in feat_imp],
    'ca_max': 30, 'midsem_max': 20,
}
with open('models/metadata.json','w') as f:
    json.dump(metadata, f, indent=2)

print("\n[6] Models saved successfully!")
print("\n" + "="*65)
print("  Training Complete on 10,000 Real Records!")
print("="*65)
