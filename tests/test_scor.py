from scor.models import *
from scor.data import *
from scor.training import *
import torch
import logging
import os
import pytest

def test_helloWorld():
    assert "Hello World!" == "Hello World!"

@pytest.fixture
def test_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return {"device": device}

@pytest.fixture
def test_dataLoading():
    train_loader, test_loader = getUnbalancedMNIST9(batch_size=32, target_class=0)
    return {"train_loader": train_loader, "test_loader": test_loader}

@pytest.fixture
def test_modelGeneration(test_device):
    device = test_device["device"]
    mlp, loss, optimizer = getMLP(device=device, loss="scor")
    resnet, _, _ = getResNet(device=device, loss="scor")
    return {"model": mlp,
            "loss": loss,
            "optimizer": optimizer}

def test_fixtures(test_device, test_dataLoading, test_modelGeneration):
    mlp, loss, optimizer = test_modelGeneration["model"], test_modelGeneration["loss"], test_modelGeneration["optimizer"]
    train_loader, test_loader = test_dataLoading["train_loader"], test_dataLoading["test_loader"]
    device = test_device["device"]

def test_directory():
    assert os.path.exists("results")

def test_emptyDirectory():
    path = "results"
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # remove file or symlink
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    assert not bool(os.listdir("results"))

def test_training(test_device):
    device = test_device["device"]
    trainTargetedMLP(batch_size=32,
                     device=device,
                     iterations=1,
                     epochs=1,)

    trainMLP(batch_size=32,
             device=device,
             iterations=1,
             epochs=1,)

    trainMLPwithdist(batch_size=32,
                     device=device,
                     iterations=1,
                     epochs=1,)

    assert os.listdir("results")  # Not such a strong test but ok
