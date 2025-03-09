from collections import OrderedDict
import pandas as pd
import numpy as np

import deepchecks.tabular.checks as checks
from deepchecks.tabular import Suite, Dataset
from deepchecks.tabular.checks import (StringMismatchComparison, TrainTestSamplesMix, NewLabelTrainTest, NewCategoryTrainTest, LabelDrift, FeatureDrift)
                                
# Define available checks (without parameters)
train_test_checkoptions = {
    "String Mismatch Comparison": StringMismatchComparison(),
    "Train Test Samples Mix": TrainTestSamplesMix(),
    "New Label Train Test": NewLabelTrainTest(),
    "New Category Train Test": NewCategoryTrainTest(),
    "Label Drift": LabelDrift(), 
    "Feature Drift": FeatureDrift(),
}

def load_csv(file_path):
    """Load CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Uploaded file is empty.")
        print(f"Loaded {file_path} with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def run_train_test(data1, data2, label, selected_tests, checks_with_params):
    """Run selected deepchecks data integrity tests on the given dataset."""
    dataset1 = Dataset(data1, label=label)
    dataset2 = Dataset(data2, label=label)
        
    # Create selected checks with conditions where applicable
    selected_checks = []
    
    for test in selected_tests:
        if test == "String Mismatch Comparison":
            check = StringMismatchComparison().add_condition_no_new_variants()
        elif test == "Train Test Samples Mix":
            check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(**checks_with_params[test])
        elif test == "New Label Train Test":
            check = NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(**checks_with_params[test])
        elif test == "New Category Train Test":
            check = NewCategoryTrainTest().add_condition_new_categories_less_or_equal(**checks_with_params[test])
        elif test == "Label Drift":
            check = LabelDrift().add_condition_drift_score_less_than(**checks_with_params[test])   
        elif test == "Feature Drift":
            check = FeatureDrift().add_condition_drift_score_less_than(**checks_with_params[test])
        else:
            check = check_options[test]

        selected_checks.append(check)
    
    #Create and run suite
    suite = Suite("Custom Train Test Evaluation Suite", *selected_checks)
    result = suite.run(dataset1, dataset2)
    # Show results
    result.show()
    result.save_as_html('train_test_evaluation_results.html')
    return None
    
