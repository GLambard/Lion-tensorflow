# :star2: Lion Optimizer for TensorFlow :star2:
This is an implementation of the Lion optimizer in TensorFlow, based on the [PyTorch implementation](https://github.com/lucidrains/lion-pytorch) 🙏 and the original [paper](https://arxiv.org/abs/2302.06675) from Xiangning Chen *et al.* 👍

# Modifications
The main modifications for the translation of the PyTorch code to TensorFlow were the replacement of PyTorch-specific functions with their TensorFlow equivalents, as well as some adjustments to match the TensorFlow API.

For example, the PyTorch `lerp_` function was replaced with the TensorFlow `tf.raw_ops.LinSpace` operator to simulate the same functionality. Additionally, the PyTorch `torch.no_grad()` context manager was replaced with the `@tf.function` decorator, which is used to wrap the `update_fn` and `step` functions to improve performance.

# Usage
Here's an example of how to use the Lion optimizer for a simple training task in TensorFlow:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lion_tensorflow import Lion

# Define a simple neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Instantiate the Lion optimizer
optimizer = Lion(lr=1e-3)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.
x_test = x_test.reshape(-1, 28*28) / 255.

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

# Reference
The Lion optimizer was introduced in the following paper:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.06675,
  doi = {10.48550/ARXIV.2302.06675},  
  url = {https://arxiv.org/abs/2302.06675}, 
  author = {Chen, Xiangning and Liang, Chen and Huang, Da and Real, Esteban and Wang, Kaiyuan and Liu, Yao and Pham, Hieu and Dong, Xuanyi and Luong, Thang and Hsieh, Cho-Jui and Lu, Yifeng and Le, Quoc V.}, 
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Computation and Language (cs.CL), Computer Vision and Pattern Recognition (cs.CV), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Symbolic Discovery of Optimization Algorithms},
  publisher = {arXiv},
  year = {2023},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

# 🚨Wanted!🚨
**Contributors and testers to this project are HIGHLY WANTED!** If you test it, find any bugs or have ideas for new features, please drop me a message,  submit an issue or submit a pull request.
