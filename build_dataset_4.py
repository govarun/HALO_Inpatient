import pickle
import numpy as np
import json
import ast
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from collections import defaultdict
import sys

mimic_dir = "/srv/local/data/varun15/mimic4/"
admissionFile = mimic_dir + "admissions.csv"
diagnosisFile = mimic_dir + "diagnoses_icd.csv"

print("Loading CSVs Into Dataframes")
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['admittime'] = pd.to_datetime(admissionDf['admittime'])
admissionDf = admissionDf.sort_values('admittime')
admissionDf = admissionDf.reset_index(drop=True)
diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("hadm_id")
diagnosisDf = diagnosisDf[diagnosisDf['icd_code'].notnull()]
diagnosisDf['icd_version_code'] = diagnosisDf['icd_version'] + "_" + diagnosisDf['icd_code']
diagnosisDf =  diagnosisDf[["icd_version_code"]]

print("Building Dataset")
data = {}
diagnosisDf_index_set = set(diagnosisDf.index)
# Pre-process diagnosisDf into a dictionary
diagnosis_dict = (diagnosisDf.groupby('hadm_id')['icd_version_code']
                  .unique()
                  .apply(list)
                  .to_dict())
# Using defaultdict to simplify data dictionary handling
data = defaultdict(lambda: {'visits': []})
all_diagnoses = set()
for row in tqdm(admissionDf.itertuples(), total=admissionDf.shape[0]):          
    #Extracting Admissions Table Info
    hadm_id = row.hadm_id
    subject_id = row.subject_id
            
    # Extracting the Diagnoses
    diagnoses = diagnosis_dict.get(hadm_id, [])
    all_diagnoses.update(diagnoses)
    # Building the hospital admission data point
    data[subject_id]['visits'].append(diagnoses)
print(len(data))

# converting the dict to a list, as the dict keys(subject_id) are no longer needed
# subject_id was earlier needed to club the visits for a single patient
data = list(data.values())

# saved all the codes present in data[x]['visits'] in a dict and the reverse dict as well
unique_codes = list(all_diagnoses)
np.random.shuffle(unique_codes)
code_to_index = {code: i for i, code in enumerate(unique_codes)}
print(f"VOCAB SIZE: {len(code_to_index)}")
index_to_code = {v: k for k, v in code_to_index.items()}

print("Adding Labels")
with open('icd_categories_groups.json', 'r') as file:
    loaded_dict = json.load(file)

icd9_categories = {ast.literal_eval(key): value for key, value in loaded_dict["icd9_categories"].items()}
icd10_categories = {ast.literal_eval(key): value for key, value in loaded_dict["icd10_categories"].items()}
group_disease_label_mapping = {ast.literal_eval(key): value for key, value in loaded_dict["group_disease_label_mapping"].items()}

# Function to classify an ICD-9 code
def classify_icd9_code(icd9_code):
    icd9_code = icd9_code[:3]
    # Handle 'E' and 'V' codes separately
    if icd9_code.startswith('E') or icd9_code.startswith('V'):
        return 16
    
    # Convert the code to a float for comparison
    try:
        code = float(icd9_code)
    except ValueError:
        return 20

    # Iterate through the map to find the matching category
    for code_range, category in icd9_categories.items():
        if code_range[0] <= code <= code_range[1]:
            return category
    
    # If no category is found
    return 20

# Function to classify an ICD-10 code
def classify_icd10_code(icd10_code):
    # Extract the letter and the numeric part of the code
    icd10_code = icd10_code[:3]

    # Iterate through the map to find the matching category
    for code_range, category in icd10_categories.items():
        if code_range[0] <= icd10_code <= code_range[1]:
            return category
    # If no category is found
    return 20


# one hot encoding for the groups that have occurred for the current row
# Add Labels
for p in tqdm(data):
  label = np.zeros(len(group_disease_label_mapping))
  for v in p['visits']:
    for c in v:
      # print(c)
      version, icd_code = c.split('_')
      if int(version) == 9:
        label[classify_icd9_code(icd_code)] = 1
      else:
        label[classify_icd10_code(icd_code)] = 1
  # break
  
  p['labels'] = label

# here we are converting the visits from the codes to the indices we generated above.
print("Converting Visits")
# converting all the codes to the indexes
for p in data:
    p['visits'] = [list(set(code_to_index[c] for c in v)) for v in p['visits']]


print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")
print(f"MAX VISIT LEN: {max([len(v) for p in data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v) for p in data for v in p['visits']])}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

# Train-Val-Test Split
print("Splitting Datasets")
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

# Save Everything
print("Saving Everything")
print(len(index_to_code))
pickle.dump(code_to_index, open("./data/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open("./data/indexToCode.pkl", "wb"))
pickle.dump(train_dataset, open("./data/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("./data/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("./data/testDataset.pkl", "wb"))