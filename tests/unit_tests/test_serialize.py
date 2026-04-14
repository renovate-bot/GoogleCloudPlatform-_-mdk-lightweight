# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test the serialization functionality of the MDK."""

import mdk.serialize
import pathlib
import unittest


class TestSerialize(unittest.TestCase):
    def test_Booster(self):
        import xgboost as xgb

        # Create a model.

        X, y = getTestData()

        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "objective": "multi:softmax",
            "max_depth": 3,
            "learning_rate": 0.1,
            "num_class": 3,
            "random_state": 42,
        }
        num_boost_round = 100
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
        )

        filename = mdk.serialize.write(bst)
        try:
            # Read it from file.
            loaded_model = mdk.serialize.read(filename)

        finally:
            # Clean up after ourselves.
            pathlib.Path(filename).unlink(missing_ok=True)

        # Verify that the loaded model matches the saved model.
        score_0 = bst.get_score()
        score_1 = loaded_model.get_score()
        for k in score_0.keys():
            self.assertEqual(score_0[k], score_1[k])

        # Do it again, but try the code path where the class is provided
        #   explicitly.
        filename = mdk.serialize.write(bst)
        try:
            # Read it from file.
            loaded_model = mdk.serialize.read(filename, model_class=xgb.Booster)

        finally:
            # Clean up after ourselves.
            pathlib.Path(filename).unlink(missing_ok=True)

        # Verify that the loaded model matches the saved model.
        score_0 = bst.get_score()
        score_1 = loaded_model.get_score()
        for k in score_0.keys():
            self.assertEqual(score_0[k], score_1[k])
        self.assertTrue(isinstance(loaded_model, xgb.Booster))

    def test_XGBClassifier(self):
        import xgboost as xgb

        # Create a model.

        X, y = getTestData()

        params = {
            "objective": "multi:softmax",
            "num_class": X.shape[1],
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)

        filename = mdk.serialize.write(model)
        try:
            # Read it from file.
            loaded_model = mdk.serialize.read(filename, model_class=xgb.XGBClassifier)

        finally:
            # Clean up after ourselves.
            pathlib.Path(filename).unlink(missing_ok=True)

        # Verify that the loaded model matches the saved model.
        self.assertTrue(isinstance(loaded_model, xgb.XGBClassifier))

    def test_None(self):
        # Set our model to None.
        model = None

        filename = mdk.serialize.write(model)
        try:
            # Read it from file.
            loaded_model = mdk.serialize.read(filename)

        finally:
            # Clean up after ourselves.
            pathlib.Path(filename).unlink(missing_ok=True)

        # Verify that the loaded model matches the saved model.
        self.assertTrue(loaded_model is None)


def getTestData():
    import sklearn.datasets

    # X, y = sklearn.datasets.make_classification(
    #     n_samples=100,
    #     n_classes=3,
    #     n_features=10,
    #     n_informative=5,
    #     random_state=42
    # )

    data = sklearn.datasets.load_iris()
    X = data["data"]
    y = data["target"]

    return X, y
