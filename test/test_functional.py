import unittest

import fish
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
import io
import sys
import os
import joblib
from fish import *
from loan import *


class FunctionalTests(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()

        # Setup for fish_is_diseased test
        try:
            df = load_fish_disease_data()
            X, y, self.df_encoded = preprocess_fish_data(df)
            X_train, X_test, y_train, y_test = split_fish_data(X, y)
            self.model = create_and_train_model(X_train, y_train)
        except:
            self.model = None
            self.df_encoded = None

    def test_load_fish_disease_data(self):
        try:
            # Expected columns in the dataset
            expected_columns = [
                'Age', 'Species', 'Water_Temperature',
                'Feeding_Behavior', 'Coloration', 'Swimming_Behavior',
                'Disease_Status'
            ]

            # Call the function
            df = load_fish_disease_data()

            # Perform checks
            is_dataframe = isinstance(df, pd.DataFrame)
            correct_length = len(df) == 1000
            has_all_columns = all(col in df.columns for col in expected_columns)

            if is_dataframe and correct_length and has_all_columns:
                self.test_obj.yakshaAssert("TestLoadFishDiseaseData", True, "functional")
                print("TestLoadFishDiseaseData = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadFishDiseaseData", False, "functional")
                print("TestLoadFishDiseaseData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadFishDiseaseData", False, "functional")
            print(f"TestLoadFishDiseaseData = Failed | Exception: {e}")

    def test_perform_eda_on_age(self):
        try:
            # Load the dataset
            df = load_fish_disease_data()

            # Check if 'Age' column exists
            if 'Age' not in df.columns:
                self.test_obj.yakshaAssert("TestPerformEDAOnAge", False, "functional")
                print("TestPerformEDAOnAge = Failed | 'Age' column missing")
                return

            # Calculate expected result
            expected_count = 860
            actual_count = df[df['Age'] > 1].shape[0]

            # Call the function (for print behavior)
            perform_eda_on_age(df)

            if actual_count == expected_count:
                self.test_obj.yakshaAssert("TestPerformEDAOnAge", True, "functional")
                print("TestPerformEDAOnAge = Passed")
            else:
                self.test_obj.yakshaAssert("TestPerformEDAOnAge", False, "functional")
                print(f"TestPerformEDAOnAge = Failed | Expected: {expected_count}, Got: {actual_count}")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPerformEDAOnAge", False, "functional")
            print(f"TestPerformEDAOnAge = Failed | Exception: {e}")

    def test_preprocess_fish_data(self):
        try:
            # Load the dataset
            df = load_fish_disease_data()

            # Call the preprocessing function
            X, y, processed_df = preprocess_fish_data(df)

            # Perform checks
            is_X_dataframe = isinstance(X, pd.DataFrame)
            is_y_series = isinstance(y, pd.Series)
            has_target_column = "Disease_Status_Healthy" in processed_df.columns
            target_removed_from_X = "Disease_Status_Healthy" not in X.columns

            if is_X_dataframe and is_y_series and has_target_column and target_removed_from_X:
                self.test_obj.yakshaAssert("TestPreprocessFishData", True, "functional")
                print("TestPreprocessFishData = Passed")
            else:
                self.test_obj.yakshaAssert("TestPreprocessFishData", False, "functional")
                print("TestPreprocessFishData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPreprocessFishData", False, "functional")
            print(f"TestPreprocessFishData = Failed | Exception: {e}")

    def test_split_fish_data(self):
        try:
            from sklearn.model_selection import train_test_split

            # Load and preprocess the dataset
            df = load_fish_disease_data()
            X, y, _ = preprocess_fish_data(df)

            # Call the split function
            X_train, X_test, y_train, y_test = split_fish_data(X, y, test_size=0.2)

            # Check if any of the returned values are None
            if X_train is None or X_test is None or y_train is None or y_test is None:
                self.test_obj.yakshaAssert("TestSplitFishData", False, "functional")
                print("TestSplitFishData = Failed | Returned None values")
                return

            # Checks
            total_records = len(X)
            expected_test_size = int(total_records * 0.2)
            expected_train_size = total_records - expected_test_size

            correct_sizes = (len(X_train) == expected_train_size) and (len(X_test) == expected_test_size)
            matching_shapes = (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == y_test.shape[0])

            if correct_sizes and matching_shapes:
                self.test_obj.yakshaAssert("TestSplitFishData", True, "functional")
                print("TestSplitFishData = Passed")
            else:
                self.test_obj.yakshaAssert("TestSplitFishData", False, "functional")
                print("TestSplitFishData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSplitFishData", False, "functional")
            print(f"TestSplitFishData = Failed | Exception: {e}")

    def test_create_and_train_model(self):
        try:
            from sklearn.tree import DecisionTreeClassifier

            # Load and preprocess the data
            df = load_fish_disease_data()
            X, y, _ = preprocess_fish_data(df)
            X_train, _, y_train, _ = split_fish_data(X, y)

            # Call the function
            model = create_and_train_model(X_train, y_train)

            # Check that it's a trained DecisionTreeClassifier
            is_model = isinstance(model, DecisionTreeClassifier)
            is_trained = hasattr(model, "tree_")  # This attribute exists only after fitting

            if is_model and is_trained:
                self.test_obj.yakshaAssert("TestCreateAndTrainModel", True, "functional")
                print("TestCreateAndTrainModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestCreateAndTrainModel", False, "functional")
                print("TestCreateAndTrainModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestCreateAndTrainModel", False, "functional")
            print(f"TestCreateAndTrainModel = Failed | Exception: {e}")

    def test_calculate_entropy(self):
        try:
            # Load and preprocess the data
            df = load_fish_disease_data()
            _, y, _ = preprocess_fish_data(df)

            # Call the entropy calculation function
            entropy = calculate_entropy(y)

            # Expected entropy (example value; update if your actual output differs)
            expected_entropy = 1.0000

            if round(entropy, 4) == expected_entropy:
                self.test_obj.yakshaAssert("TestCalculateEntropy", True, "functional")
                print("TestCalculateEntropy = Passed")
            else:
                self.test_obj.yakshaAssert("TestCalculateEntropy", False, "functional")
                print(f"TestCalculateEntropy = Failed | Expected: {expected_entropy}, Got: {round(entropy, 4)}")
        except Exception as e:
            self.test_obj.yakshaAssert("TestCalculateEntropy", False, "functional")
            print(f"TestCalculateEntropy = Failed | Exception: {e}")

    def test_fish_is_diseased(self):
        try:
            # Call the function to check the prediction for new fish data
            result = fish.check_new_data_from_json(self.model, df_encoded=self.df_encoded,
                                                   json_file="fish_data.json")

            # Check if the predicted result is 0 (i.e., diseased) and model is properly trained
            if result == 0 and self.model is not None and self.df_encoded is not None:
                self.test_obj.yakshaAssert("TestFishIsDiseased", True, "functional")
                print("TestFishIsDiseased = Passed")
            else:
                self.test_obj.yakshaAssert("TestFishIsDiseased", False, "functional")
                print("TestFishIsDiseased = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestFishIsDiseased", False, "functional")
            print(f"TestFishIsDiseased = Failed: {str(e)}")

    def test_load_data(self):
        try:
            # Expected column names
            expected_columns = [
                'loan_amount', 'term', 'credit_score', 'employment_length',
                'home_ownership', 'annual_income', 'defaulted'
            ]

            # Load the data
            df = load_data("loan_dataset.csv")

            # Check if it's a DataFrame and all expected columns exist
            has_all_columns = all(col in df.columns for col in expected_columns)
            correct_mean = round(df['loan_amount'].mean(), 2) == 15096.66
            correct_max = round(df['loan_amount'].max(), 2) == 34263.66

            if isinstance(df, pd.DataFrame) and has_all_columns and correct_mean and correct_max:
                self.test_obj.yakshaAssert("TestLoadData", True, "functional")
                print("TestLoadData = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadData", False, "functional")
                print("TestLoadData = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadData", False, "functional")
            print(f"TestLoadData = Failed | Exception: {e}")

    def test_explore_home_ownership(self):
        try:
            # Load the dataset
            df = load_data("loan_dataset.csv")

            # Call the function to test
            rent_count = explore_home_ownership(df)

            # Check if the returned value is None
            if rent_count is None:
                self.test_obj.yakshaAssert("TestExploreHomeOwnership", False, "functional")
                print("TestExploreHomeOwnership = Failed | Returned None value")
                return

            # Expected result
            expected_count = 339

            if rent_count == expected_count:
                self.test_obj.yakshaAssert("TestExploreHomeOwnership", True, "functional")
                print("TestExploreHomeOwnership = Passed")
            else:
                self.test_obj.yakshaAssert("TestExploreHomeOwnership", False, "functional")
                print("TestExploreHomeOwnership = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestExploreHomeOwnership", False, "functional")
            print(f"TestExploreHomeOwnership = Failed | Exception: {e}")

    def test_prepare_data(self):
        try:
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import numpy as np

            # Load original data
            df = load_data("loan_dataset.csv")

            # Capture original dtypes for label-encoded columns
            original_term_dtype = df['term'].dtype
            original_home_dtype = df['home_ownership'].dtype

            # Prepare the data
            processed_df = prepare_data(df.copy())

            # Check that result is a DataFrame
            is_dataframe = isinstance(processed_df, pd.DataFrame)

            # Check that term and home_ownership were encoded only if they were object type
            is_term_encoded = (
                    original_term_dtype == 'object' and np.issubdtype(processed_df['term'].dtype, np.number)
            )
            is_home_encoded = (
                    original_home_dtype == 'object' and np.issubdtype(processed_df['home_ownership'].dtype, np.number)
            )

            # Check that features are scaled properly
            scaled_columns = processed_df.columns.difference(['defaulted'])
            means = processed_df[scaled_columns].mean().round(1)
            stds = processed_df[scaled_columns].std().round(1)
            scaling_correct = all(means.abs() <= 0.1) and all((stds - 1).abs() <= 0.1)

            if is_dataframe and is_term_encoded and is_home_encoded and scaling_correct:
                self.test_obj.yakshaAssert("TestPrepareData", True, "functional")
                print("TestPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
                print("TestPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
            print(f"TestPrepareData = Failed | Exception: {e}")

    def test_sigmoid_demo(self):
        try:
            # Call the function
            result = sigmoid_demo()

            # Check if the returned value is None
            if result is None:
                self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
                print("TestSigmoidDemo = Failed | Returned None value")
                return

            # Expected output
            expected_value = 0.8176

            # Check if result is close enough (4 decimal precision)
            if round(result, 4) == expected_value:
                self.test_obj.yakshaAssert("TestSigmoidDemo", True, "functional")
                print("TestSigmoidDemo = Passed")
            else:
                self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
                print(f"TestSigmoidDemo = Failed | Expected: {expected_value}, Got: {round(result, 4)}")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
            print(f"TestSigmoidDemo = Failed | Exception: {e}")

    def test_train_and_evaluate(self):
        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            import os

            # Load and preprocess the data
            df = load_data("loan_dataset.csv")
            df = prepare_data(df)

            # Split data into features and target
            X = df.drop("defaulted", axis=1)
            y = df["defaulted"]

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Call the function
            result = train_and_evaluate(X_train, y_train, X_test, y_test, path="loan_model.pkl")

            # Check if the returned value is None
            if result is None:
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
                print("TestTrainAndEvaluate = Failed | Returned None value")
                return

            model = result.get("model")
            y_pred = result.get("y_pred")
            y_prob = result.get("y_pred_prob")

            # Check model file
            file_exists = os.path.exists("loan_model.pkl")

            # Check model and outputs
            has_model = isinstance(model, LogisticRegression)
            has_y_pred = isinstance(y_pred, np.ndarray)
            has_y_prob = isinstance(y_prob, np.ndarray)

            # Check model attributes
            has_coef = hasattr(model, "coef_")
            has_intercept = hasattr(model, "intercept_")
            has_classes = hasattr(model, "classes_")

            if all([file_exists, has_model, has_y_pred, has_y_prob, has_coef, has_intercept, has_classes]):
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", True, "functional")
                print("TestTrainAndEvaluate = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
                print("TestTrainAndEvaluate = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
            print(f"TestTrainAndEvaluate = Failed | Exception: {e}")


if __name__ == '__main__':
    unittest.main()
