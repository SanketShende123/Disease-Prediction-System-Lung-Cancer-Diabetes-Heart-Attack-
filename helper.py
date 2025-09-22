import pandas as pd
import numpy as np

df = pd.read_csv('data/clean_dataset.tsv', sep='\t')

# Ensure correct feature selection
if "Disease" in df.columns:  
    df = df.drop(columns=["Disease"])  # Drop target column if present

def prepare_symptoms_array(symptoms):
    '''
    Convert a list of symptoms into an input feature array for the ML model.

    Output:
    - X (np.array) = 1D array of length equal to the number of symptoms in the dataset
    '''
    num_features = len(df.columns)  # Dynamically determine the number of features
    symptoms_array = np.zeros((1, num_features))

    for symptom in symptoms:
        if symptom in df.columns:  
            symptom_idx = df.columns.get_loc(symptom)
            symptoms_array[0, symptom_idx] = 1
        else:
            print(f"Warning: '{symptom}' not found in dataset.")

    return symptoms_array

# Verify array shape before using it in ML
X = prepare_symptoms_array(["cough", "high_fever"])
print("Prepared input shape:", X.shape)  # Ensure this is (1, 133)
