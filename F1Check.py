from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import TrainTestCheck, Dataset
from sklearn.metrics import f1_score

class F1Check(TrainTestCheck):
    """Custom Deepchecks check for evaluating F1 with or without a model."""

    def __init__(self, use_model=True, y_train_pred=None, y_test_pred=None, **kwargs):
        """
        use_model: If True, use the model for prediction. If False, rely on provided predicted labels.
        y_train_pred: Predicted labels for train set (if use_model=False).
        y_test_pred: Predicted labels for test set (if use_model=False).
        """
        super().__init__(**kwargs)
        self.use_model = use_model  # Control whether to use the model
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred

    def run_logic(self, context) -> CheckResult:
        """Computes F1 score for both train and test datasets."""
        train_dataset = context.train
        test_dataset = context.test
        model = context.model if self.use_model else None  # Use model only if enabled

        if train_dataset is None or test_dataset is None:
            return CheckResult(False, display="Train and test datasets are required.")

        if train_dataset.label_name is None or test_dataset.label_name is None:
            return CheckResult(False, display="Both train and test datasets must have actual labels.")

        # Extract true labels
        y_train_true = train_dataset.data[train_dataset.label_name].values
        y_test_true = test_dataset.data[test_dataset.label_name].values

        # Get predictions: Either from the model or from provided predicted labels
        if self.use_model:
            if model is None:
                return CheckResult(False, display="Model is required but not provided.")
            y_train_pred = model.predict(train_dataset.data[train_dataset.features])
            y_test_pred = model.predict(test_dataset.data[test_dataset.features])
        else:
            if self.y_train_pred is None or self.y_test_pred is None:
                return CheckResult(False, display="Predicted labels for train and test must be provided when use_model=False.")
            y_train_pred = self.y_train_pred
            y_test_pred = self.y_test_pred

        # Compute F1
        train_F1 = f1_score(y_train_true, y_train_pred, average='binary')
        test_F1 = f1_score(y_test_true, y_test_pred, average='binary')

        display_str = (f'Train F1: {train_F1:.4f} \n'
                       f'Test F1: {test_F1:.4f}')

        return CheckResult(value={"train_F1": train_F1, "test_F1": test_F1}, display=display_str)

    def add_condition_F1_above(self, threshold: float = 0.80):
        """Adds a condition to ensure F1 is above a given threshold."""

        def condition(result_value):
            train_F1 = result_value["train_F1"]
            test_F1 = result_value["test_F1"]

            train_status = "PASS" if train_F1 >= threshold else "FAIL"
            test_status = "PASS" if test_F1 >= threshold else "FAIL"

            message = (f"Train F1: {train_F1:.4f} ({train_status})\n"
                       f"Test F1: {test_F1:.4f} ({test_status})")

            if train_F1 >= threshold and test_F1 >= threshold:
                return ConditionResult(ConditionCategory.PASS, message)
            else:
                return ConditionResult(ConditionCategory.FAIL, message)

        return self.add_condition(f'F1 ≥ {threshold} for both train & test', condition)


# Import dependencies for testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Simulate predicted labels (e.g., from a pre-saved model output)
y_train_pred = np.random.choice([0, 1], size=len(y_train))  # Simulated train predictions
y_test_pred = np.random.choice([0, 1], size=len(y_test))  # Simulated test predictions

# Convert train and test sets to Deepchecks Dataset
train_dataset = Dataset(X_train, label=y_train)
test_dataset = Dataset(X_test, label=y_test)

# Option 1: Run the check using a model
F1_check_model = F1Check(use_model=True).add_condition_F1_above(0.80)
result_model = F1_check_model.run(train_dataset, test_dataset, model=model)
result_model.show()

# Option 2: Run the check using only predicted labels (no model)
F1_check_no_model = F1Check(use_model=False, y_train_pred=y_train_pred, y_test_pred=y_test_pred)
F1_check_no_model.add_condition_F1_above(0.80)
result_no_model = F1_check_no_model.run(train_dataset, test_dataset)
result_no_model.show()
