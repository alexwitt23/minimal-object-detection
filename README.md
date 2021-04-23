# Tire Detector

Eveything needed to train an an object detector in PyTorch with python3.

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


## Example

[Here](https://user-images.githubusercontent.com/31543169/115086127-4e10a800-9ed1-11eb-8d0c-919587538381.jpg)
is an example image.


Update the image path in the script, then run the main script like so:

```
PYTHONPATH=. python src/main.py
```


## Python3 vs Python2

I haven't tested this in python2. We would have to try building PyTorch for python2. We could also look at using C++.
