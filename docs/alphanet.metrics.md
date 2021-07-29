<!-- markdownlint-disable -->

<a href="../src/alphanet/metrics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `alphanet.metrics`
训练中的辅助准确率信息. 

该module包含涨跌精度信息的计算类。 使用到该类的模型在保存后重新读取时需要添加``custom_objects``参数， 或者使用``alphanet.load_model()``函数。 



---

<a href="../src/alphanet/metrics.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UpDownAccuracy`
通过对return的预测来计算涨跌准确率. 

<a href="../src/alphanet/metrics.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(name='up_down_accuracy', **kwargs)
```

涨跌准确率. 


---

#### <kbd>property</kbd> activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> dtype





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

<a href="../src/alphanet/metrics.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset_state`

```python
reset_state()
```

在train、val的epoch末尾重置精度. 

---

<a href="../src/alphanet/metrics.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `result`

```python
result()
```

获取结果. 

---

<a href="../src/alphanet/metrics.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_state`

```python
update_state(y_true, y_pred, sample_weight=None)
```

加入新的预测更新精度值. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
