# AlphaNet

![unittest](https://github.com/Congyuwang/AlphaNetV3/actions/workflows/tests.yml/badge.svg)
[![Congyuwang](https://circleci.com/gh/Congyuwang/AlphaNetV3.svg?style=shield)](https://circleci.com/gh/Congyuwang/AlphaNetV3)
![publish](https://github.com/Congyuwang/AlphaNetV3/actions/workflows/python-publish.yml/badge.svg)

## A Recurrent Neural Network For Predicting Stock Prices

### AlphaNetV2

Below is the structure of AlphaNetV2

```
input: (batch_size, history time steps, features)

                 stride = 5
input -> expand features -> BN -> LSTM -> BN -> Dense(linear)
```

### AlphaNetV3

Below is the structure of AlphaNetV3

```
input: (batch_size, history time steps, features)

                 stride = 5
        +-> expand features -> BN -> GRU -> BN -+
input --|       stride = 10                     |- concat -> Dense(linear)
        +-> expand features -> BN -> GRU -> BN -+
```

## Installation
Either clone this repository or just use pypi: ``pip install alphanet``.

The pypi project is here: [alphanet](https://pypi.org/project/alphanet/).

## Documentation
For detailed documentation, go to
[alphanet documentation](https://github.com/Congyuwang/AlphaNetV3/tree/master/docs).
