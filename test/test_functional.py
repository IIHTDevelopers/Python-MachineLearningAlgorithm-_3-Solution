import unittest
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
import io
import sys
import os
import joblib
from House import load_and_preprocess, show_key_stats, prepare_data, train_and_save_model, evaluate_model
from loan import load_and_prepare_data, explore_data, sigmoid_demo, cost_function, train_and_evaluate


class TestHouse(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()
        
        # Prepare test data for House.py
        self.features = ['rooms', 'area', 'bathrooms', 'floors', 'age']
        self.target = 'price'

    def test_load_and_preprocess(self):
        """
        Test case for load_and_preprocess() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            df = load_and_preprocess("Housing.csv")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is loaded correctly
            expected_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                               'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                               'parking', 'prefarea', 'furnishingstatus']
            
            if (isinstance(df, pd.DataFrame) and 
                all(col.lower() in df.columns for col in ['price', 'rooms', 'area', 'bathrooms', 'floors', 'age']) and
                "✅ Data loaded and cleaned." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", True, "functional")
                print("TestLoadAndPreprocess = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
                print("TestLoadAndPreprocess = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
            print(f"TestLoadAndPreprocess = Failed | Exception: {e}")

    def test_show_key_stats(self):
        """
        Test case for show_key_stats() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Load data and call the function
            df = load_and_preprocess("Housing.csv")
            show_key_stats(df)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if stats are displayed correctly
            output = captured_output.getvalue()
            if ("Standard Deviation of Price" in output and 
                "Maximum Number of Rooms" in output):
                self.test_obj.yakshaAssert("TestShowKeyStats", True, "functional")
                print("TestShowKeyStats = Passed")
            else:
                self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
                print("TestShowKeyStats = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestShowKeyStats", False, "functional")
            print(f"TestShowKeyStats = Failed | Exception: {e}")

    def test_prepare_data(self):
        """
        Test case for prepare_data() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Load data and call the function
            df = load_and_preprocess("Housing.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is prepared correctly
            if (isinstance(X_train, np.ndarray) and 
                isinstance(X_test, np.ndarray) and 
                (isinstance(y_train, pd.Series) or isinstance(y_train, np.ndarray)) and 
                (isinstance(y_test, pd.Series) or isinstance(y_test, np.ndarray)) and
                "Data prepared and split." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestPrepareData", True, "functional")
                print("TestPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
                print("TestPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPrepareData", False, "functional")
            print(f"TestPrepareData = Failed | Exception: {e}")

    def test_train_and_save_model(self):
        """
        Test case for train_and_save_model() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Load data, prepare it, and call the function
            df = load_and_preprocess("Housing.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)
            model = train_and_save_model(X_train, y_train, "test_house_model.pkl")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if model is trained and saved correctly
            if (os.path.exists("test_house_model.pkl") and 
                "Model trained and saved" in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", True, "functional")
                print("TestTrainAndSaveModel = Passed")
                # Clean up
                if os.path.exists("test_house_model.pkl"):
                    os.remove("test_house_model.pkl")
            else:
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
                print("TestTrainAndSaveModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
            print(f"TestTrainAndSaveModel = Failed | Exception: {e}")

    def test_evaluate_model(self):
        """
        Test case for evaluate_model() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Load data, prepare it, train model, and call the function
            df = load_and_preprocess("Housing.csv")
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, self.features, self.target)
            model = train_and_save_model(X_train, y_train, "test_house_model.pkl")
            evaluate_model(model, X_test, y_test)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if model evaluation is displayed correctly
            output = captured_output.getvalue()
            if ("Mean Squared Error" in output and 
                "Sample Predictions" in output):
                self.test_obj.yakshaAssert("TestEvaluateModel", True, "functional")
                print("TestEvaluateModel = Passed")
                # Clean up
                if os.path.exists("test_house_model.pkl"):
                    os.remove("test_house_model.pkl")
            else:
                self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print("TestEvaluateModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
            print(f"TestEvaluateModel = Failed | Exception: {e}")


class TestLoan(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()

    def test_load_and_prepare_data(self):
        """
        Test case for load_and_prepare_data() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            df = load_and_prepare_data("loan_dataset.csv")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is loaded and prepared correctly
            if (isinstance(df, pd.DataFrame) and 
                'defaulted' in df.columns and
                "Real dataset loaded and preprocessed." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", True, "functional")
                print("TestLoadAndPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
                print("TestLoadAndPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPrepareData", False, "functional")
            print(f"TestLoadAndPrepareData = Failed | Exception: {e}")

    def test_explore_data(self):
        """
        Test case for explore_data() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Load data and call the function
            df = load_and_prepare_data("loan_dataset.csv")
            explore_data(df)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Since explore_data is a pass function, just check if it runs without errors
            self.test_obj.yakshaAssert("TestExploreData", True, "functional")
            print("TestExploreData = Passed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestExploreData", False, "functional")
            print(f"TestExploreData = Failed | Exception: {e}")

    def test_sigmoid_demo(self):
        """
        Test case for sigmoid_demo() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            sigmoid_demo()
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if sigmoid value is displayed correctly
            if "Sigmoid(1.5) = " in captured_output.getvalue():
                self.test_obj.yakshaAssert("TestSigmoidDemo", True, "functional")
                print("TestSigmoidDemo = Passed")
            else:
                self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
                print("TestSigmoidDemo = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSigmoidDemo", False, "functional")
            print(f"TestSigmoidDemo = Failed | Exception: {e}")

    def test_cost_function(self):
        """
        Test case for cost_function() function.
        """
        try:
            # Create test data
            y_true = np.array([0, 1, 0, 1])
            y_pred_prob = np.array([0.1, 0.9, 0.2, 0.8])
            
            # Call the function
            cost = cost_function(y_true, y_pred_prob)
            
            # Check if cost is calculated correctly
            if isinstance(cost, float) and cost > 0:
                self.test_obj.yakshaAssert("TestCostFunction", True, "functional")
                print("TestCostFunction = Passed")
            else:
                self.test_obj.yakshaAssert("TestCostFunction", False, "functional")
                print("TestCostFunction = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestCostFunction", False, "functional")
            print(f"TestCostFunction = Failed | Exception: {e}")

    def test_train_and_evaluate(self):
        """
        Test case for train_and_evaluate() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Load data, prepare it, and call the function
            df = load_and_prepare_data("loan_dataset.csv")
            X = df.drop(columns=['defaulted'])
            y = df['defaulted']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            train_and_evaluate(X_train, y_train, X_test, y_test, "test_loan_model.pkl")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if model is trained and evaluated correctly
            output = captured_output.getvalue()
            if (os.path.exists("test_loan_model.pkl") and 
                "Model trained and saved" in output and
                "Log Loss (Custom Cost)" in output and
                "Sample Predictions" in output):
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", True, "functional")
                print("TestTrainAndEvaluate = Passed")
                # Clean up
                if os.path.exists("test_loan_model.pkl"):
                    os.remove("test_loan_model.pkl")
            else:
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
                print("TestTrainAndEvaluate = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
            print(f"TestTrainAndEvaluate = Failed | Exception: {e}")


if __name__ == '__main__':
    unittest.main()
