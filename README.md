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

## Example

### Step 0: import alphanet
```python
from alphanet import AlphaNetV3, load_model
from alphanet.data import TrainValData, TimeSeriesData
from alphanet.metrics import UpDownAccuracy
```

### Step 1: build data
```python
# read data
df = pd.read_csv("some_data.csv")

# compute label (future return)
df_future_return = here_you_compute_it_by_your_self
df = df_future_return.merge(df,
                            how="inner",
                            left_on=["date", "security_code"],
                            right_on=["date", "security_code"])

# create an empty list
stock_data_list = []

# put each stock into the list using TimeSeriesData() class
security_codes = df["security_code"].unique()
for code in security_codes:
    table_part = df.loc[df["security_code"] == code, :]
    stock_data_list.append(TimeSeriesData(dates=table_part["date"].values,                   # date column
                                          data=table_part.iloc[:, 3:].values,                # data columns
                                          labels=table_part["future_10_cum_return"].values)) # label column

# put stock list into TrainValData() class, specify dataset lengths
train_val_data = TrainValData(time_series_list=stock_data_list,
                              train_length=1200,   # 1200 trading days for training
                              validate_length=150, # 150 trading days for validation
                              history_length=30,   # each input contains 30 days of history
                              sample_step=2,       # jump to days forward for each sampling
                              train_val_gap=10     # leave a 10-day gap between training and validation
```

### Step 2: get datasets from desired period
```python
# get one training period that start from 20110131
train, val, dates_info = data_producer.get(20110131, order="by_date")
print(dates_info)
```

### Step 3: compile the model and start training
```python
# get an AlphaNetV3 instance
model = AlphaNetV3(l2=0.001, dropout=0.0)

# you may use UpDownAccuracy() here to evaluate performance
model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                       UpDownAccuracy()]

# train
model.fit(train.batch(500).cache(),
          validation_data=val.batch(500).cache(),
          epochs=100)
```

### Step 4: save and load

#### saving
```python
# save model by save method
model.save("path_to_your_model")

# or just save weights
model.save_weights("path_to_your_weights")
```

#### loading
```python
# load entire model using load_model() from alphanet module
model = load_model("path_to_your_model")

# only load weights by first creating a model instance
model = AlphaNetV3(l2=0.001, dropout=0.0)
model.load_weights("path_to_your_weights")
```

Note: only `alphanet.load_model(filename)` recognizes custom `UpDownAccuracy`.
If you do not use `UpDownAccuracy`,
you can _also_ use `tf.keras.models.load_model(filename)`.

## Documentation
For detailed documentation, go to
[alphanet documentation](https://github.com/Congyuwang/AlphaNetV3/tree/master/docs).

For implementation details, go to
[alphanet source folder](https://github.com/Congyuwang/AlphaNetV3/tree/master/src).

### One Little Caveat
The model expands features quadratically.
So, if you have 5 features, it will be expanded to more than 50 features (for AlphaNetV3),
and if you have 10 features, it will be expanded to more than 200 features.
Therefore, do not put too many features inside.

### One More Note
``alphanet.data``module is completely independent from ``alphanet`` module,
and can be a useful tool for training any timeseries neural network.
