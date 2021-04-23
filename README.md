# Tire Detector

Eveything needed to train an object detector with PyTorch and python3.

## Setup

All that's needed is to install the requirements with:

```
python3 -m pip install -r requirements.txt
```

You might also choose to install in a virtual environment:

```
python3 -m venv ~/.envs/minimal-od
source ~/.envs/minimal-od/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Models

The object detector here is a [RetinaNet](https://arxiv.org/abs/1708.02002) model.

The backbone of the model is a Resnet variant. We opt to use the
Retinanet-18 variant instead of the heavier options because we are
only training the model with one class. The other models are still
available.

## Training

Check out the [training instructions](/src/train/README.md).
