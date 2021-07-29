# AlphaNetV3

![unittest](https://github.com/Congyuwang/AlphaNetV3/actions/workflows/tests.yml/badge.svg)
[![Congyuwang](https://circleci.com/gh/Congyuwang/AlphaNetV3.svg?style=shield)](https://circleci.com/gh/Congyuwang/AlphaNetV3)
![publish](https://github.com/Congyuwang/AlphaNetV3/actions/workflows/python-publish.yml/badge.svg)

## A Recurrent Neural Network For Predicting Stock Prices

Requirement: tensorflow 2.

1. The model and data utility are in `./alphanet/` folder.
2. CSI500 and CSI800 stock market data are included in `./data/` folder.
3. Model Configuration is in `config.py`.
4. To run the model, execute `python main.py`.
5. The loss plot pictures and models will be stored by each epoch and training period in `./models/` folder.
6. Tests of layers and time series data tools are in `./tests/` folder.
   To run the test, execute `python -m unittest tests.tests`.

## Installation
Either clone this repository or just use pypi:
``pip install alphanet``.

The pypi project is here: https://pypi.org/project/alphanet/.
