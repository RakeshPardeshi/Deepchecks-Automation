import pickle
import deepchecks
import pandas as pd
from deepchecks.tabular import Suite
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    TrainTestPerformance, RocReport, SimpleModelComparison, ConfusionMatrixReport, UnusedFeatures, 
    WeakSegmentsPerformance, PredictionDrift, BoostingOverfit
)

# Define available checks (without parameters)
#model_checkoptions = {
#    "Train Test Performance": TrainTestPerformance(),
#    "Roc Report": RocReport(),
#    "Simple Model Comparison": SimpleModelComparison(),
#    "Confusion Matrix Report": ConfusionMatrixReport(),
#}

def load_model_from_pickle(filepath='model.pkl'):
    """Load a trained model from a pickle file."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filepath}")
    return model


def run_model_evaluation(data1, data2, label, selected_tests, checks_with_params, prediction_label_column, prediction_proba_col, model_file):

    if prediction_label_column is not None:
        y_pred_train_1lod = data1.pop(prediction_label_column)
        y_pred_test_1lod = data2.pop(prediction_label_column)
        y_proba_train_1lod = data1.pop(prediction_proba_col)
        y_proba_test_1lod = data2.pop(prediction_proba_col)
    
    
    train_dataset = Dataset(data1, label = label)
    test_dataset = Dataset(data2, label = label)
      
    # Create selected checks with conditions where applicable
    selected_checks = []
    for test in selected_tests:
        if test == "Train Test Performance":
            check = TrainTestPerformance().add_condition_test_performance_greater_than(**checks_with_params[test])
        elif test == "Roc Report":
            check = RocReport().add_condition_auc_greater_than(**checks_with_params[test])
        elif test == "Simple Model Comparison":
            check = SimpleModelComparison()
        elif test == "Confusion Matrix Report":
            check = ConfusionMatrixReport()
        elif test == "Unused Features":
            check = UnusedFeatures(feature_importance_threshold = 0.01 ).add_condition_number_of_high_variance_unused_features_less_or_equal( **checks_with_params[test])
        elif test == "Weak Segments Performance":
            check = WeakSegmentsPerformance(**checks_with_params[test])
        elif test == "Prediction Drift":
            check = PredictionDrift(drift_mode = 'proba').add_condition_drift_score_less_than(**checks_with_params[test]) 
        elif test == "Boosting Overfit":
            check = BoostingOverfit().add_condition_test_score_percent_decline_less_than(**checks_with_params[test]) 
        selected_checks.append(check)
    
    #Run moedl evaluation checks
    if (prediction_label_column is not None) & (y_proba_train_1lod is not None):

        trained_model = load_model_from_pickle(filepath=model_file)
        
        #Create and run suite
        suite1 = Suite("Custom Model Evaluation Suite - Predictions", *selected_checks)
        result1 = suite1.run(train_dataset = train_dataset, 
                            test_dataset = test_dataset, 
                            y_pred_train = y_pred_train_1lod, 
                            y_pred_test = y_pred_test_1lod, 
                            y_proba_train = y_proba_train_1lod,
                            y_proba_test = y_proba_test_1lod)
        result1.show()
        result1.save_as_html('Model_Eval_Results - Predictions.html')
        
    if model_file is not None:
        trained_model = load_model_from_pickle(filepath=model_file)
        feat_list = trained_model.feature_names_in_
        feat_list = list(trained_model.feature_names_in_)
        #
        #print(feat_list)
        #display(train_dataset)
        y_pred_train = trained_model.predict_proba(train_dataset.data[feat_list])
        y_pred_test = trained_model.predict_proba(test_dataset.data[feat_list])
        print(type(y_pred_train))
        print(y_pred_train.dtype)
        print(y_pred_train)

        feat_list.append(label)
        train_dataset = Dataset(train_dataset.data[feat_list], label=label)
        test_dataset = Dataset(test_dataset.data[feat_list], label=label)
        #display(train_dataset)
        suite2 = Suite("Custom Model Evaluation Suite - Model File", *selected_checks)
        result2 = suite2.run(train_dataset = train_dataset, 
                             test_dataset = test_dataset, 
                             y_pred_train = y_pred_train,
                             y_pred_test = y_pred_test,
                             model = trained_model)
        result2.show()
        result2.save_as_html('Model_Eval_Results - Trained Model File.html')
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


