# Deep-Learning-Hardware-Benchmark
This repository contains my proposed implementation for benchmarking in order to evaluate whether a setup of hardware is feasible for deep learning projects. 

## Materials and Methods
1. To evaluate the hardware performance, the following metrics are used:
  * The **total execution time** (training + validation time) of a deep learning model on a dataset.
  * (Optional) The **total inference time** (model loading + prediction time) of a deep learning on a test dataset.
  * The **presence of potential computing resources error** including but not limited to the Out-Of-Memory (OOM) error.

2. I use the deep learning models from my Master Thesis for thg benchmark purpose: the first model is a modified VGG19 based on a study by Deitsch et al., and the other is a modified concatenated model first proposed in a study from Rahimzadeh et al.

3. I use the following datasets for the benchmark.
* The **original MNIST dataset** developed by Yann LeCun et al.

![](images/mnist.png)

* The **Zalando MNIST dataset** developed by Han Xiao.

![](images/zalando.png)

4. Please refer the `Instruction Guide.ipynb` to install Python, Tensorflow, and other dependencies.

## Results
To provide a solid baseline for comparison, I benchmarked my own computing resources and recorded the results. Below is the information of my hardware:

Component | Description
| ------------ | ------------- |
| CPU | Intel(R) Core(TM) i7-10750H @2.60 GHz |
| GPU | NVIDIA GeForce RTX 2060 5980 MB |
| Memory | 16384 MB RAM |
| Storage | 476 GB |

The results of my benchmark are as follows:

Test | Execution Time (s) | Inference Time (s) | Error
| ------------ | ------------- | ------------- | ------------- |
|Original MNIST | 3553 | No error |
| Fashion MNIST | 3770 | No error |


# References
<a id="1">[1]</a> 
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion,O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vander-plas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay,“Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

<a id="2">[2]</a>
F. Chollet, “Keras,” 2015. https://github.com/fchollet/keras

<a id ="3">[3]</a>
Xiao, K. Rasul, and R. Vollgraf, “A Novel Image Dataset for Benchmarking Machine Learning Algorithms,” 2017. https://github.com/zalandoresearch/fashion-mnist
