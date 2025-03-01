import deepchecks
import pandas as pd
from deepchecks.tabular import Suite
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    IsSingleValue, SpecialCharacters, MixedNulls, MixedDataTypes, StringMismatch, DataDuplicates,
    StringLengthOutOfBounds, ConflictingLabels, OutlierSampleDetection, FeatureLabelCorrelation,
    FeatureFeatureCorrelation, IdentifierLabelCorrelation
)

# Define available checks (without parameters)
check_options = {
    "Is Single Value": IsSingleValue(),
    "Special Characters": SpecialCharacters(),
    "Mixed Nulls": MixedNulls(),
    "Mixed Data Types": MixedDataTypes(),
    "String Mismatch": StringMismatch(),  # Needs parameter
    "Data Duplicates": DataDuplicates(),
    "String Length Out Of Bounds": StringLengthOutOfBounds(),
    "Conflicting Labels": ConflictingLabels(),
    "Outlier Sample Detection": OutlierSampleDetection(),  # Needs parameter
    "Feature Label Correlation": FeatureLabelCorrelation(),
    "Feature Feature Correlation": FeatureFeatureCorrelation(),  # Needs parameter
    "Identifier Label Correlation": IdentifierLabelCorrelation()
}


def run_data_integrity(data1, data2, label, selected_tests, checks_with_params):
    """Run selected deepchecks data integrity tests on the given dataset."""
    dataset1 = Dataset(data1, label=label)
    dataset2 = Dataset(data2, label=label)
        
    # Create selected checks with conditions where applicable
    selected_checks = []
    for test in selected_tests:
        if test == "String Mismatch":
            check = StringMismatch().add_condition_number_variants_less_or_equal(**checks_with_params[test])
        elif test == "Outlier Sample Detection":
            ignore_columns = []
            if label is not None:
                ignore_columns.append(label)
            check = OutlierSampleDetection(ignore_columns = [label]).add_condition_outlier_ratio_less_or_equal(**checks_with_params[test])
        elif test == "Feature Feature Correlation":
            check = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(**checks_with_params[test],  
                n_pairs = 0)
        elif test == "Is Single Value":
            check = IsSingleValue().add_condition_not_single_value()
        elif test == "Special Characters":
            check = SpecialCharacters().add_condition_ratio_of_special_characters_less_or_equal(**checks_with_params[test])   
        elif test == "Mixed Nulls":
            check = MixedNulls().add_condition_different_nulls_less_equal_to(**checks_with_params[test])
        elif test == "Mixed Data Types":
            check = MixedDataTypes().add_condition_rare_type_ratio_not_in_range(**checks_with_params[test])
        elif test == "Data Duplicates":
            check = DataDuplicates().add_condition_ratio_less_or_equal(**checks_with_params[test])
        elif test == "Conflicting Labels":
            print(test)
            check = ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal(**checks_with_params[test])
        elif test == "Feature Label Correlation":
            check = FeatureLabelCorrelation().add_condition_feature_pps_less_than(**checks_with_params[test])
        elif test == "Identifier Label Correlation":
            check = IdentifierLabelCorrelation().add_condition_pps_less_or_equal(**checks_with_params[test])
        elif test == "String Length Out Of Bounds":
            check = StringLengthOutOfBounds().add_condition_number_of_outliers_less_or_equal(**checks_with_params[test])
        else:
            check = check_options[test]

        selected_checks.append(check)
    
    # Create and run suite
    suite = Suite("Custom Data Integrity Suite", *selected_checks)
    result = suite.run(dataset1, dataset2)

    # Show results
    result.show()
    result.save_as_html('data_integrity_results.html')
    return None


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


