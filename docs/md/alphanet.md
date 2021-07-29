<!-- markdownlint-disable -->

<a href="../../src/alphanet/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `alphanet`
复现华泰金工 alpha net V3 版本. 

```
input: (batch_size, history time steps, features)

                 stride = 5
         +-> expand features -> BN -> GRU -> BN -+
input --|       stride = 10                     |- concat -> Dense(linear)
         +-> expand features -> BN -> GRU -> BN -+
``` 

(BN: batch normalization) 

version: 0.0.4 

author: Congyu Wang 

date: 2021-07-29 

该module定义了计算不同时间序列特征的层，工程上使用tensorflow 进行高度向量化的计算，训练时较高效。 

**Global Variables**
---------------
- **metrics**

---

<a href="../../src/alphanet/__init__.py#L603"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_model`

```python
load_model(
    filepath,
    custom_objects: dict = None,
    compile: bool = True,
    options=None
)
```

包装``tf.kreas``的``load_model``，添加``UpDownAccuracy``. 



**Args:**
 
 - <b>`filepath`</b>:  文件路径: 
        - String or `pathlib.Path` object, path to the saved model 
        - `h5py.File` object from which to load the model 
 - <b>`custom_objects`</b>:  自定义类的识别，从类或函数名到类或函数的映射字典. 
 - <b>`compile`</b>:  Boolean, 是否compile model. 
 - <b>`options`</b>:  其他 `tf.saved_model.LoadOptions`. 



**Returns:**
 A Keras model instance. If the original model was compiled, and saved with the optimizer, then the returned model will be compiled. Otherwise, the model will be left uncompiled. In the case that an uncompiled model is returned, a warning is displayed if the `compile` argument is set to `True`. 



**Raises:**
 
 - <b>`ImportError`</b>:  if loading from an hdf5 file and h5py is not available. 
 - <b>`IOError`</b>:  In case of an invalid savefile 


---

<a href="../../src/alphanet/__init__.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Std`
每个序列各个stride的标准差. 



**Notes:**

> 计算每个feature各个stride的standard deviation 

<a href="../../src/alphanet/__init__.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(stride: int = 10, **kwargs)
```

标准差. 



**Args:**
 
 - <b>`stride`</b> (int):  time steps需要是stride的整数倍 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

List of metrics added using the `add_metric()` API. 



**Example:**
 

``` input = tf.keras.layers.Input(shape=(3,))```
``` d = tf.keras.layers.Dense(2)``` ``` output = d(input)```
``` d.add_metric(tf.reduce_max(output), name='max')``` ``` d.add_metric(tf.reduce_min(output), name='min')```
``` [m.name for m in d.metrics]``` ['max', 'min'] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights

List of all non-trainable weights tracked by this layer. 

Non-trainable weights are *not* updated during training. They are expected to be updated manually in `call()`. 



**Returns:**
  A list of non-trainable variables. 

---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights

List of all trainable weights tracked by this layer. 

Trainable weights are updated via gradient descent during training. 



**Returns:**
  A list of trainable variables. 

---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build`

```python
build(input_shape)
```

构建该层，计算维度信息. 

---

<a href="../../src/alphanet/__init__.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, *args, **kwargs)
```

函数主逻辑实现部分. 



**Args:**
 
 - <b>`inputs`</b> (tensor):  输入dimension为(batch_size, time_steps, features) 



**Returns:**
 dimension 为(batch_size, time_steps / stride, features) 

---

<a href="../../src/alphanet/__init__.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 


---

<a href="../../src/alphanet/__init__.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ZScore`
每个序列各个stride的均值除以其标准差. 



**Notes:**

> 并非严格意义上的z-score, 计算公式为每个feature各个stride的mean除以各自的standard deviation 

<a href="../../src/alphanet/__init__.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(stride: int = 10, **kwargs)
```

均值除以标准差. 



**Args:**
 
 - <b>`stride`</b> (int):  time steps需要是stride的整数倍 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

List of metrics added using the `add_metric()` API. 



**Example:**
 

``` input = tf.keras.layers.Input(shape=(3,))```
``` d = tf.keras.layers.Dense(2)``` ``` output = d(input)```
``` d.add_metric(tf.reduce_max(output), name='max')``` ``` d.add_metric(tf.reduce_min(output), name='min')```
``` [m.name for m in d.metrics]``` ['max', 'min'] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights

List of all non-trainable weights tracked by this layer. 

Non-trainable weights are *not* updated during training. They are expected to be updated manually in `call()`. 



**Returns:**
  A list of non-trainable variables. 

---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights

List of all trainable weights tracked by this layer. 

Trainable weights are updated via gradient descent during training. 



**Returns:**
  A list of trainable variables. 

---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build`

```python
build(input_shape)
```

构建该层，计算维度信息. 

---

<a href="../../src/alphanet/__init__.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, *args, **kwargs)
```

函数主逻辑实现部分. 



**Args:**
 
 - <b>`inputs`</b> (tensor):  输入dimension为(batch_size, time_steps, features) 



**Returns:**
 dimension 为(batch_size, time_steps / stride, features) 

---

<a href="../../src/alphanet/__init__.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 


---

<a href="../../src/alphanet/__init__.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearDecay`
每个序列各个stride的线性衰减加权平均. 



**Notes:**

> 以线性衰减为权重，计算每个feature各个stride的均值： 如stride为10，则某feature该stride的权重为(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) 

<a href="../../src/alphanet/__init__.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(stride=10, **kwargs)
```

线性递减加权平均. 



**Args:**
 
 - <b>`stride`</b> (int):  time steps需要是stride的整数倍 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

List of metrics added using the `add_metric()` API. 



**Example:**
 

``` input = tf.keras.layers.Input(shape=(3,))```
``` d = tf.keras.layers.Dense(2)``` ``` output = d(input)```
``` d.add_metric(tf.reduce_max(output), name='max')``` ``` d.add_metric(tf.reduce_min(output), name='min')```
``` [m.name for m in d.metrics]``` ['max', 'min'] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights

List of all non-trainable weights tracked by this layer. 

Non-trainable weights are *not* updated during training. They are expected to be updated manually in `call()`. 



**Returns:**
  A list of non-trainable variables. 

---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights

List of all trainable weights tracked by this layer. 

Trainable weights are updated via gradient descent during training. 



**Returns:**
  A list of trainable variables. 

---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build`

```python
build(input_shape)
```

构建该层，计算维度信息. 

---

<a href="../../src/alphanet/__init__.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, *args, **kwargs)
```

函数主逻辑实现部分. 



**Args:**
 
 - <b>`inputs`</b> (tensor):  输入dimension为(batch_size, time_steps, features) 



**Returns:**
 dimension 为(batch_size, time_steps / stride, features) 

---

<a href="../../src/alphanet/__init__.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 


---

<a href="../../src/alphanet/__init__.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Return`
每个序列各个stride的回报率. 



**Notes:**

> 计算公式为每个stride最后一个数除以第一个数再减去一 

<a href="../../src/alphanet/__init__.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(stride=10, **kwargs)
```

回报率. 



**Args:**
 
 - <b>`stride`</b> (int):  time steps需要是stride的整数倍 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

List of metrics added using the `add_metric()` API. 



**Example:**
 

``` input = tf.keras.layers.Input(shape=(3,))```
``` d = tf.keras.layers.Dense(2)``` ``` output = d(input)```
``` d.add_metric(tf.reduce_max(output), name='max')``` ``` d.add_metric(tf.reduce_min(output), name='min')```
``` [m.name for m in d.metrics]``` ['max', 'min'] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights

List of all non-trainable weights tracked by this layer. 

Non-trainable weights are *not* updated during training. They are expected to be updated manually in `call()`. 



**Returns:**
  A list of non-trainable variables. 

---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights

List of all trainable weights tracked by this layer. 

Trainable weights are updated via gradient descent during training. 



**Returns:**
  A list of trainable variables. 

---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build`

```python
build(input_shape)
```

构建该层，计算维度信息. 

---

<a href="../../src/alphanet/__init__.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, *args, **kwargs)
```

函数主逻辑实现部分. 



**Args:**
 
 - <b>`inputs`</b> (tensor):  输入dimension为(batch_size, time_steps, features) 



**Returns:**
 dimension 为(batch_size, time_steps / stride, features) 

---

<a href="../../src/alphanet/__init__.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 


---

<a href="../../src/alphanet/__init__.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Covariance`
每个stride各个时间序列的covariance. 



**Notes:**

> 计算每个stride每两个feature之间的covariance大小， 输出feature数量为features * (features - 1) / 2 

<a href="../../src/alphanet/__init__.py#L298"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(stride=10, **kwargs)
```

外乘类的扩张层. 



**Args:**
 
 - <b>`stride`</b> (int):  time steps需要是stride的整数倍 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

List of metrics added using the `add_metric()` API. 



**Example:**
 

``` input = tf.keras.layers.Input(shape=(3,))```
``` d = tf.keras.layers.Dense(2)``` ``` output = d(input)```
``` d.add_metric(tf.reduce_max(output), name='max')``` ``` d.add_metric(tf.reduce_min(output), name='min')```
``` [m.name for m in d.metrics]``` ['max', 'min'] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights

List of all non-trainable weights tracked by this layer. 

Non-trainable weights are *not* updated during training. They are expected to be updated manually in `call()`. 



**Returns:**
  A list of non-trainable variables. 

---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights

List of all trainable weights tracked by this layer. 

Trainable weights are updated via gradient descent during training. 



**Returns:**
  A list of trainable variables. 

---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build`

```python
build(input_shape)
```

构建该层，计算维度信息. 

---

<a href="../../src/alphanet/__init__.py#L344"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, *args, **kwargs)
```

函数主逻辑实现部分. 



**Args:**
 
 - <b>`inputs`</b> (tensor):  输入dimension为(batch_size, time_steps, features) 



**Returns:**
 dimension 为(batch_size, time_steps / stride, features * (features - 1) / 2) 

---

<a href="../../src/alphanet/__init__.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 


---

<a href="../../src/alphanet/__init__.py#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Correlation`
每个stride各个时间序列的相关系数. 



**Notes:**

> 计算每个stride每两个feature之间的correlation coefficient， 输出feature数量为features * (features - 1) / 2 

<a href="../../src/alphanet/__init__.py#L298"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(stride=10, **kwargs)
```

外乘类的扩张层. 



**Args:**
 
 - <b>`stride`</b> (int):  time steps需要是stride的整数倍 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

List of metrics added using the `add_metric()` API. 



**Example:**
 

``` input = tf.keras.layers.Input(shape=(3,))```
``` d = tf.keras.layers.Dense(2)``` ``` output = d(input)```
``` d.add_metric(tf.reduce_max(output), name='max')``` ``` d.add_metric(tf.reduce_min(output), name='min')```
``` [m.name for m in d.metrics]``` ['max', 'min'] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights

List of all non-trainable weights tracked by this layer. 

Non-trainable weights are *not* updated during training. They are expected to be updated manually in `call()`. 



**Returns:**
  A list of non-trainable variables. 

---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights

List of all trainable weights tracked by this layer. 

Trainable weights are updated via gradient descent during training. 



**Returns:**
  A list of trainable variables. 

---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build`

```python
build(input_shape)
```

构建该层，计算维度信息. 

---

<a href="../../src/alphanet/__init__.py#L391"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, *args, **kwargs)
```

函数主逻辑实现部分. 



**Args:**
 
 - <b>`inputs`</b> (tensor):  输入dimension为(batch_size, time_steps, features) 



**Returns:**
 dimension 为(batch_size, time_steps / stride, features * (features - 1) / 2) 

---

<a href="../../src/alphanet/__init__.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 


---

<a href="../../src/alphanet/__init__.py#L440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureExpansion`
时间序列特征扩张层. 



**Notes:**

> 该层扩张时间序列的feature数量，并通过stride缩短时间序列长度， 其包括一下一些feature: 
>- standard deviation 
>- mean / standard deviation 
>- linear decay average 
>- return of each stride 
>- covariance of each two features for each stride 
>- correlation coefficient of each two features for each stride 

<a href="../../src/alphanet/__init__.py#L461"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(stride=10, **kwargs)
```

时间序列特征扩张. 



**Args:**
 
 - <b>`stride`</b> (int):  time steps需要是stride的整数倍 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

List of metrics added using the `add_metric()` API. 



**Example:**
 

``` input = tf.keras.layers.Input(shape=(3,))```
``` d = tf.keras.layers.Dense(2)``` ``` output = d(input)```
``` d.add_metric(tf.reduce_max(output), name='max')``` ``` d.add_metric(tf.reduce_min(output), name='min')```
``` [m.name for m in d.metrics]``` ['max', 'min'] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights

List of all non-trainable weights tracked by this layer. 

Non-trainable weights are *not* updated during training. They are expected to be updated manually in `call()`. 



**Returns:**
  A list of non-trainable variables. 

---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights

List of all trainable weights tracked by this layer. 

Trainable weights are updated via gradient descent during training. 



**Returns:**
  A list of trainable variables. 

---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L480"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, *args, **kwargs)
```

函数主逻辑实现部分. 



**Args:**
 
 - <b>`inputs`</b> (tensor):  输入dimension为(batch_size, time_steps, features) 



**Returns:**
 dimension 为(batch_size, time_steps / stride, features * (features + 3)) 

---

<a href="../../src/alphanet/__init__.py#L504"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 


---

<a href="../../src/alphanet/__init__.py#L511"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlphaNetV3`
alpha net v3版本模型. 



**Notes:**

> 复现华泰金工 alpha net V3 版本 :
``` 
>input: (batch_size, history time steps, features) 
>stride = 5 +-> expand -> BN -> GRU -> BN -+ input --|       stride = 10            |- concat -> Dense(linear) +-> expand -> BN -> GRU -> BN -+ 
>(BN: batch normalization) 

<a href="../../src/alphanet/__init__.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dropout=0.0, l2=0.001, *args, **kwargs)
```

Alpha net v3. 



**Notes:**

> alpha net v3 版本的全tensorflow实现，结构详见代码展开 
>

**Args:**
 
 - <b>`dropout`</b>:  跟在特征扩张以及Batch Normalization之后的dropout，默认无dropout 
 - <b>`l2`</b>:  输出层的l2-regularization参数 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> distribute_strategy

The `tf.distribute.Strategy` this model was created under. 

---

#### <kbd>property</kbd> dtype

Dtype used by the weights of the layer, set in the constructor. 

---

#### <kbd>property</kbd> dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> inbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> layers





---

#### <kbd>property</kbd> losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` model.add_loss(tf.abs(tf.reduce_mean(x)))``` ``` model.losses```
[<tf.Tensor 'Abs:0' shape=() dtype=float32>]

``` inputs = tf.keras.Input(shape=(10,))``` ``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')```
``` x = d(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Weight regularization.```
``` model.add_loss(lambda: tf.reduce_mean(d.kernel))``` ``` model.losses```
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]



**Returns:**

   A list of tensors.


---

#### <kbd>property</kbd> metrics

Returns the model's metrics added using `compile`, `add_metric` APIs. 

Note: Metrics passed to `compile()` are available only after a `keras.Model` has been trained/evaluated on actual data. 



**Examples:**
 

``` inputs = tf.keras.layers.Input(shape=(3,))```
``` outputs = tf.keras.layers.Dense(2)(inputs)``` ``` model = tf.keras.models.Model(inputs=inputs, outputs=outputs)```
``` model.compile(optimizer="Adam", loss="mse", metrics=["mae"])``` ``` [m.name for m in model.metrics]```
[]

``` x = np.random.random((2, 3))``` ``` y = np.random.randint(0, 2, (2, 2))```
``` model.fit(x, y)``` ``` [m.name for m in model.metrics]```
['loss', 'mae']

``` inputs = tf.keras.layers.Input(shape=(3,))``` ``` d = tf.keras.layers.Dense(2, name='out')```
``` output_1 = d(inputs)``` ``` output_2 = d(inputs)```
``` model = tf.keras.models.Model(``` ...    inputs=inputs, outputs=[output_1, output_2]) ``` model.add_metric(```
...    tf.reduce_sum(output_2), name='mean', aggregation='mean')
``` model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])``` ``` model.fit(x, (y, y))```
``` [m.name for m in model.metrics]``` ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae', 'out_1_acc', 'mean'] 

---

#### <kbd>property</kbd> metrics_names

Returns the model's display labels for all outputs. 

Note: `metrics_names` are available only after a `keras.Model` has been trained/evaluated on actual data. 



**Examples:**
 

``` inputs = tf.keras.layers.Input(shape=(3,))```
``` outputs = tf.keras.layers.Dense(2)(inputs)``` ``` model = tf.keras.models.Model(inputs=inputs, outputs=outputs)```
``` model.compile(optimizer="Adam", loss="mse", metrics=["mae"])``` ``` model.metrics_names```
[]

``` x = np.random.random((2, 3))``` ``` y = np.random.randint(0, 2, (2, 2))```
``` model.fit(x, y)``` ``` model.metrics_names```
['loss', 'mae']

``` inputs = tf.keras.layers.Input(shape=(3,))``` ``` d = tf.keras.layers.Dense(2, name='out')```
``` output_1 = d(inputs)``` ``` output_2 = d(inputs)```
``` model = tf.keras.models.Model(``` ...    inputs=inputs, outputs=[output_1, output_2]) ``` model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])```
``` model.fit(x, (y, y))``` ``` model.metrics_names```
['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
'out_1_acc']


---

#### <kbd>property</kbd> name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> non_trainable_variables





---

#### <kbd>property</kbd> non_trainable_weights





---

#### <kbd>property</kbd> outbound_nodes

Deprecated, do NOT use! Only for compatibility with external Keras. 

---

#### <kbd>property</kbd> output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> run_eagerly

Settable attribute indicating whether the model should run eagerly. 

Running eagerly means that your model will be run step by step, like Python code. Your model might run slower, but it should become easier for you to debug it by stepping into individual layer calls. 

By default, we will attempt to compile your model to a static graph to deliver the best execution performance. 



**Returns:**
  Boolean, whether the model should run eagerly. 

---

#### <kbd>property</kbd> state_updates

Deprecated, do NOT use! (deprecated) 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

Returns the `updates` from all layers that are stateful. 

This is useful for separating training updates and state updates, e.g. when we need to update a layer's internal state during prediction. 



**Returns:**
  A list of update ops. 

---

#### <kbd>property</kbd> stateful





---

#### <kbd>property</kbd> submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> trainable





---

#### <kbd>property</kbd> trainable_variables





---

#### <kbd>property</kbd> trainable_weights





---

#### <kbd>property</kbd> updates

DEPRECATED FUNCTION 

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. 

---

#### <kbd>property</kbd> variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> weights

Returns the list of all layer variables/weights. 



**Returns:**
  A list of variables. 



---

<a href="../../src/alphanet/__init__.py#L563"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(inputs, training=None, mask=None)
```

计算逻辑实现. 

---

<a href="../../src/alphanet/__init__.py#L579"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    optimizer=<tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7fd464e46e20>,
    loss='MSE',
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    **kwargs
)
```

设置优化器、loss、metric等. 

---

<a href="../../src/alphanet/__init__.py#L595"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```

获取参数，保存模型需要的函数. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
