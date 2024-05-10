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

# load data and remove data with extreme outlier coordinates or negative fares
data = pd.read_csv("data/medical_data_train.csv")
y = data.readmitted

base_features = [
    "number_inpatient", "num_medications", "number_diagnoses", "num_lab_procedures",
    "num_procedures", "time_in_hospital", "number_outpatient", "number_emergency",
    "gender_Female", "payer_code_?", "medical_specialty_?", "diag_1_428", "diag_1_414",
    "diabetesMed_Yes", "A1Cresult_None"
]

X = data[base_features].astype(float)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# For speed, we will calculate shap values on smaller subset of the validation data
small_val_X = val_X.iloc[:150]
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)

"""
The first few questions require examining the distribution of effects for each feature, rather than just an average effect for each feature. 
"""

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(small_val_X)

shap.summary_plot(shap_values[:, :, 1], small_val_X, show=False)
fname = "results/Advanced_SHAP_Values_Exercise_HospitalData_SummaryPlot"
plt.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close()

"""
QUESTION 1

Which of the following features has a bigger range of effects on predictions (i.e. larger difference between most positive and most negative effect)
    - diag_1_428 or
    - payer_code_?

diag_1_428 has a bigger range


QUESTION 2

Do you believe the range of effects sizes (distance between smallest effect and largest effect) 
is a good indication of which feature will have a higher permutation importance? Why or why not?

If the range of effect sizes measures something different from permutation importance: which is a better answer for the question 
"Which of these two features does the model say is more important for us to understand when discussing readmission risks in the population?"

No. The width of the effects range is not a reasonable approximation to permutation importance. 
For that matter, the width of the range doesn't map well to any intuitive sense of "importance" because it can be determined by just a few outliers. 
However if all dots on the graph are widely spread from each other, that is a reasonable indication that permutation importance is high.
Because the range of effects is so sensitive to outliers, permutation importance is a better measure of what's generally important to the model.


QUESTION 3

Both diag_1_428 and payer_code_? are binary variables, taking values of 0 or 1.

From the graph, which do you think would typically have a bigger impact on predicted readmission risk:
    - Changing diag_1_428 from 0 to 1
    - Changing payer_code_? from 0 to 1

Changing diag_1_428. While most SHAP values of diag_1_428 are small, the few pink dots 
(high values of the variable, corresponding to people with that diagnosis) have large SHAP values. 
In other words, the pink dots for this variable are far from 0, and making someone have the higher (pink) value would increase their readmission risk significantly. 
In real-world terms, this diagnosis is rare, but poses a larger risk for people who have it.
In contrast, payer_code_? has many values of both blue and pink, and both have SHAP values that differ meaningfully from 0.
But changing payer_code_? from 0 (blue) to 1 (pink) is likely to have a smaller impact than changing diag_1_428.

QUESTION 4

Some features (like number_inpatient) have reasonably clear separation between the blue and pink dots.
Other variables like num_lab_procedures have blue and pink dots jumbled together, even though the SHAP values (or impacts on prediction) aren't all 0.

What do you think you learn from the fact that num_lab_procedures has blue and pink dots jumbled together?

The jumbling suggests that sometimes increasing that feature leads to higher predictions, and other times it leads to a lower prediction. 
Said another way, both high and low values of the feature can have both positive and negative effects on the prediction.
The most likely explanation for this "jumbling" of effects is that the variable (in this case num_lab_procedures) has an interaction effect with other variables.
For example, there may be some diagnoses for which it is good to have many lab procedures,
and other diagnoses where suggests increased risk. 
We don't yet know what other feature is interacting with num_lab_procedures though we could investigate that with SHAP contribution dependence plots.


QUESTION 5

Both num_medications and num_lab_procedures share that jumbling of pink and blue dots.

Aside from num_medications having effects of greater magnitude (both more positive and more negative),
it's hard to see a meaningful difference between how these two features affect readmission risk. 
Create the SHAP dependence contribution plots for each variable, and describe what you think is different between how these two variables affect predictions.
"""

# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(small_val_X)

features_to_plot = ["num_medications", "num_lab_procedures"]
for feat in features_to_plot:
    shap.dependence_plot(feat, shap_values[:, :, 1], small_val_X, interaction_index=feat, show=False)

    fname = f"results/Advanced_SHAP_Values_Exercise_HospitalData_{feat}_DependencePlot"
    plt.savefig(fname, bbox_inches="tight")
    print(f"{fname} written")
    plt.close()

"""
Loosely speaking, num_lab_procedures looks like a cloud with little disernible pattern. 
It does not slope steeply up nor down at any point. It's hard to say we've learned much from that plot.
At the same time, the values are not all very close to 0. So the model seems to think this is a relevant feature.
One potential next step would be to explore more by coloring it with different other features to search for an interaction.

On the other hand, num_medications clearly slopes up until a value of about 20, and then it turns back down. 
Without more medical background, this seems a surprising phenomenon... 
You could do some exploration to see whether these patients have unusual values for other features too.
But a good next step would be to discuss this phenomenon with domain experts (in this case, the doctors).
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
