"""
resnet, train based on dataset, freeze parameters and weights, then use the trained resnet as the embedding network
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import (
    resnet18,  # smallest resnet since our images are small (10x10)
)


def train_resnet(x_train_img, x_test_img, theta_train, theta_test, batch_size, device):
    resnet = PS2DResNet()

    #batch_size = 512 #bad habit, should be passed in parameters to be the same as the normalizing flow
    max_epochs = 300

    criterion = torch.nn.MSELoss()  # mean square error loss for the regression task
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-3)

    resnet.to(device)

    train_dataset = TensorDataset(x_train_img, theta_train)
    val_dataset = TensorDataset(
        x_test_img, theta_test
    )  # test is acting as val here in this case
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # shuffling
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )  # no shuffling

    best_val_loss = float("inf")
    best_weights = None
    epochs_without_better_loss = 0  # after 10 epochs with no improvement, we stop

    for epoch in range(max_epochs):
        print(f"epoch number {epoch}/{max_epochs}")
        resnet.train()  # train mode on (this means dropout is active) -> dropout should NOT be active after training when evaluating since the normalizing flow wants something it can go back and forth on
        train_loss = 0.0

        for x_batch, theta_batch in train_loader:
            x_batch, theta_batch = (
                x_batch.to(device),
                theta_batch.to(device),
            )

            optimizer.zero_grad()  # get rid of old gradients
            predictions = resnet(x_batch)
            loss = criterion(predictions, theta_batch)  # get loss of pred vs real
            loss.backward()  # backward propagation, according to the loss metric
            optimizer.step()  # this will update weigths, according to the optimizer

            train_loss += loss.item() * x_batch.size(0)  # batch loss

        train_loss /= len(train_loader.dataset)  # total loss of all batches

        resnet.eval()  # evaluation mode so we can do validation loss
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, theta_batch in val_loader:
                x_batch, theta_batch = (
                    x_batch.to(device),
                    theta_batch.to(device),
                )

                predictions = resnet(x_batch)
                loss = criterion(predictions, theta_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(
                resnet.state_dict()
            )  # deepcopy since this will change every epoch
            epochs_without_better_loss = 0
        else:
            epochs_without_better_loss += 1

        if epochs_without_better_loss >= 15:
            break  # early stopping

    resnet.load_state_dict(best_weights)
    print("val loss: ", best_val_loss)
    resnet.eval()  # set to ecval mode so dropout is off when used by sbi
    for param in resnet.parameters():
        param.requires_grad = False  # freeze weigts

    return resnet


class PS2DResNet(nn.Module):
    """
    Embedding network is based on the resnet-18 architecture but modified to suit the small image matrix
    """
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=None)  # we want to train the weights ourselves
        self.resnet.conv1 = nn.Conv2d(
            in_channels=3,  # the three 3 redshift bins
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )  # small-ish kernel size and stride + padding since we only have 10x10 to work with, this is a modification of the first layer of the resnet

        self.resnet.maxpool = nn.Identity()  # this is basically a pass through because per default resnet uses max pooling but 10x10 is too small for that

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 64)
        self.end_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),  # Gelu not needed here
            nn.Linear(64, 3),  # down to 3, for the three redshift bins
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.end_layer(x)
        return x
