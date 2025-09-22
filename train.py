import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import joblib
import gzip

# Import dataset
dataset_df = pd.read_csv('data/dataset.csv')

# Preprocess
dataset_df = dataset_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
dataset_df.fillna(0, inplace=True)  # Handle missing values

# One-hot encoding for symptoms
symptom_df = pd.get_dummies(dataset_df.filter(regex='Symptom'), prefix='', prefix_sep='')
symptom_df = symptom_df.T.groupby(symptom_df.T.index).agg("max").T

# Merge with Disease column
clean_df = pd.concat([symptom_df, dataset_df['Disease']], axis=1)

# Save cleaned dataset
clean_df.to_csv('data/clean_dataset.tsv', sep='\t', index=False)

# Prepare features and labels
X_data = clean_df.iloc[:, :-1]
y_data = clean_df.iloc[:, -1].astype('category')

# Encode labels
le = preprocessing.LabelEncoder()
y_data = le.fit_transform(y_data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(eval_metric='mlogloss')


# Train model
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, preds)
print(f"The accuracy of the model is {accuracy:.4f}")

# Save model and label encoder
joblib.dump(model, gzip.open('model/model_binary.dat.gz', "wb"))
model.save_model("model/xgboost_model.json")
joblib.dump(le, gzip.open('model/label_encoder.dat.gz', "wb"))
