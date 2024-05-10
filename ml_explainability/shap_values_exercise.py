import time
tic = time.time()

from pdb import set_trace

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
import shap

"""
THE SCENARIO

A hospital has struggled with "readmissions," where they release a patient before the patient has recovered enough,
and the patient returns with health complications.

The hospital wants your help identifying patients at highest risk of being readmitted. 
Doctors (rather than your model) will make the final decision about when to release each patient,
but they hope your model will highlight issues the doctors should consider when releasing a patient.
"""

# load data and remove data with extreme outlier coordinates or negative fares
data = pd.read_csv("data/medical_data_train.csv")

"""
The hospital has given you relevant patient medical information. Here is a list of columns in the data:

data.columns

Index(['time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'race_Caucasian',
       'race_AfricanAmerican', 'gender_Female', 'age_[70-80)', 'age_[60-70)',
       'age_[50-60)', 'age_[80-90)', 'age_[40-50)', 'payer_code_?',
       'payer_code_MC', 'payer_code_HM', 'payer_code_SP', 'payer_code_BC',
       'medical_specialty_?', 'medical_specialty_InternalMedicine',
       'medical_specialty_Emergency/Trauma',
       'medical_specialty_Family/GeneralPractice',
       'medical_specialty_Cardiology', 'diag_1_428', 'diag_1_414',
       'diag_1_786', 'diag_2_276', 'diag_2_428', 'diag_2_250', 'diag_2_427',
       'diag_3_250', 'diag_3_401', 'diag_3_276', 'diag_3_428',
       'max_glu_serum_None', 'A1Cresult_None', 'metformin_No',
       'repaglinide_No', 'nateglinide_No', 'chlorpropamide_No',
       'glimepiride_No', 'acetohexamide_No', 'glipizide_No', 'glyburide_No',
       'tolbutamide_No', 'pioglitazone_No', 'rosiglitazone_No', 'acarbose_No',
       'miglitol_No', 'troglitazone_No', 'tolazamide_No', 'examide_No',
       'citoglipton_No', 'insulin_No', 'glyburide-metformin_No',
       'glipizide-metformin_No', 'glimepiride-pioglitazone_No',
       'metformin-rosiglitazone_No', 'metformin-pioglitazone_No', 'change_No',
       'diabetesMed_Yes', 'readmitted'],
      dtype='object')

Here are some quick hints at interpreting the field names:
    - Your prediction target is readmitted
    - Columns with the word 'diag' indicate the diagnostic code of the illness or illnesses the patient was admitted with.
    For example, diag_1_428 means the doctor said their first illness diagnosis is number "428".
    What illness does 428 correspond to? You could look it up in a codebook, but without more medical background it wouldn't mean anything to you anyway.
    - Column names like 'glimepiride_No' mean the patient did not have the medicine glimepiride. 
    If this feature had a value of False, then the patient did take the drug glimepiride
    - Features whose names begin with 'medical_specialty' describe the specialty of the doctor seeing the patient. The values in these fields are all True or False.


STEP 1

You have built a simple model, but the doctors say they don't know how to evaluate a model, 
and they'd like you to show them some evidence the model is doing something in line with their medical intuition. 
Create any graphics or tables that will show them a quick overview of what the model is doing?

They are very busy. So they want you to condense your model overview into just 1 or 2 graphics, rather than a long string of graphics.
"""

y = data.readmitted

base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)

# make and show permutation importances
perm = permutation_importance(my_model, val_X, val_y, random_state=1)
sorted_importances_idx = (-perm.importances_mean).argsort()
importances = pd.DataFrame(
    np.stack((perm.importances_mean[sorted_importances_idx], perm.importances_std[sorted_importances_idx])).T,
    index=val_X.columns[sorted_importances_idx],
    columns=["Mean", "Std"],
)

print(importances)
"""
print(importances)

                          Mean       Std
number_inpatient      0.046336  0.004682
number_emergency      0.011936  0.001686
number_outpatient     0.006848  0.002304
number_diagnoses      0.006304  0.001665
num_medications       0.002880  0.004708
...                        ...       ...
payer_code_MC        -0.001568  0.001639
race_AfricanAmerican -0.001728  0.001440
diag_3_401           -0.001760  0.001657
age_[40-50)          -0.002176  0.001282
gender_Female        -0.002720  0.002613


STEP 2

It appears number_inpatient is a really important feature. The doctors would like to know more about that. 
Create a graph for them that shows how num_inpatient affects the model's predictions.
"""

# make partial dependence plots for most important feature as determined by permutation_importance
feat_to_plot = val_X.columns[sorted_importances_idx][0]

fig, ax = plt.subplots()
PartialDependenceDisplay.from_estimator(my_model, val_X, [feat_to_plot], ax=ax)
fname = f"results/SHAP_Values_Exercise_RandForestClass_{feat_to_plot}_PartialPlot"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
STEP 3

The doctors think it's a good sign that increasing the number of inpatient procedures leads to increased predictions. 
But they can't tell from this plot whether that change in the plot is big or small.
They'd like you to create something similar for time_in_hospital to see how that compares.
"""

# make partial dependence plots for most important feature as determined by permutation_importance
feat_to_plot = "time_in_hospital"

fig, ax = plt.subplots()
PartialDependenceDisplay.from_estimator(my_model, val_X, [feat_to_plot], ax=ax)
fname = f"results/SHAP_Values_Exercise_RandForestClass_{feat_to_plot}_PartialPlot"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
STEP 4

It seems like time_in_hospital doesn't matter at all. The difference between the lowest value on the partial dependence plot and the highest value is about 5%.

If that is what your model concluded, the doctors will believe it. But it seems so low. Could the data be wrong, or is your model doing something more complex than they expect?

They'd like you to show them the raw readmission rate for each value of time_in_hospital to see how it compares to the partial dependence plot.
"""

# combine all training data
all_training = pd.concat([train_X, train_y], axis=1)
grouped_train = all_training.groupby(["time_in_hospital"]).mean()["readmitted"]

fig, ax = plt.subplots()
grouped_train.plot(ax=ax)
ax.set(ylabel="Readmission Rate")

fname = "results/SHAP_Values_Exercise_Time_In_Hospital_vs Readmitted"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
STEP 5

Now the doctors are convinced you have the right data, and the model overview looked reasonable.
It's time to turn this into a finished product they can use. Specifically, the hospital wants you to create a function patient_risk_factors that does the following:
    1) Takes a single row with patient data (of the same format you as your raw data)
    2) Creates a visualization showing what features of that patient increased their risk of readmission, what features decreased it, and how much those features mattered.

It's not important to show every feature with every miniscule impact on the readmission risk. It's fine to focus on only the most important features for that patient.
"""

def patient_risk_factors(model, X, patient_num):
    data_for_prediction = X.iloc[patient_num]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    explainer = shap.TreeExplainer(model)
    
    # Calculate Shap values
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[:,1], data_for_prediction, matplotlib=True, show=False)


row_to_show = 5
fig = patient_risk_factors(my_model, val_X, row_to_show)

fname = f"results/SHAP_Values_Exercise_Patient{row_to_show}_SHAPvals"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
