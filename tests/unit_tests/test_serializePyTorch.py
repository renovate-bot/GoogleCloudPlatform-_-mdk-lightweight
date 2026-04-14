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

"""Test the serialization functionality of the MDK for a PyTorch model."""

import mdk.serialize
from torch import nn
import pathlib
import unittest


class TestSerializPyTorch(unittest.TestCase):
    def test_PyTorch(self):
        import torch
        import numpy as np

        # Create a model.

        X, y = getTestData()

        print(f"{X.shape = }")
        num_classes = len(set(y))
        num_features = X.shape[1]

        print(f"{num_classes = }")
        print(f"{num_features = }")

        model = NeuralNetworkClassificationModel(num_features, num_classes)

        # Set up our input and output data format.
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # Set up our parameters.
        learning_rate = 0.01
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        num_epochs = 1000
        train_losses = np.zeros(num_epochs)
        test_losses = np.zeros(num_epochs)

        train_network(
            model, optimizer, criterion, X, y, num_epochs, train_losses, test_losses
        )

        filename = mdk.serialize.write(model)
        try:
            # This context manager addresses security concerns with pickle.
            with torch.serialization.safe_globals(
                [
                    NeuralNetworkClassificationModel,
                    torch.nn.modules.linear.Linear,
                    torch.nn.modules.activation.ReLU,
                ]
            ):
                # Read the model from file.
                loaded_model = mdk.serialize.read(filename)

        finally:
            # Clean up after ourselves.
            pathlib.Path(filename).unlink(missing_ok=True)

        print(f"{type(loaded_model) = }")

        # Verify that the loaded model matches the saved model.
        self.assertTrue(isinstance(loaded_model, NeuralNetworkClassificationModel))


class NeuralNetworkClassificationModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        x,
    ):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out


def train_network(
    model,
    optimizer,
    criterion,
    X_train,
    y_train,
    num_epochs,
    train_losses,
    test_losses,
):
    for epoch in range(num_epochs):
        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        # forward feed
        output_train = model(X_train)

        # calculate the loss
        loss_train = criterion(output_train, y_train)

        # backward propagation: calculate gradients
        loss_train.backward()

        # update the weights
        optimizer.step()

        train_losses[epoch] = loss_train.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f},"
            )


def getTestData():
    import sklearn.datasets

    data = sklearn.datasets.load_iris()
    X = data["data"]
    y = data["target"]

    return X, y
