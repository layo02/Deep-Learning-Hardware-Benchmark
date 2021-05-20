# Deep-Learning-Hardware-Benchmark
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction
This repository contains the proposed implementation for benchmarking to evaluate whether a setup of hardware is feasible for complex deep learning projects.

## Scope
* The benchmark evaluates the performance of a setup having a single CPU, a single GPU, RAM and memory storage. The performance of a multi-CPUs/multi-GPUs or server-based is not considered.
* The benchmark is built on the **Anaconda** distribution of Python, and the **Jupyter Notebook** IDE. The deep learning models mentioned in this benchmarked are implemented uisng Keras.

## Evaluation metrics:
To evaluate the performance, the following metrics are used:
1. **Total execution time**: the **total execution time** includes both the **total training time** and the **total validation time** of a deep learning model on a dataset after a defined number of epochs. Here, the number of epochs is 100. The lower the **total execution time** the better.
2. **Total inference time**: the **total inference time** includes both the **model loading time** (the time required to fully load a set of pre-trained weights to implement a model) and the **total prediction time** of a deep learning model on a test dataset. Similar to the **total execution time**, the lower the **total inference time** the better.
3. **FLOPS**: the performance capability of a CPU or GPU can be measured by counting the number of floating operation points (FLO) it can execute per second. Thus, the higher the **FLOPS**, the better. 
4. **Computing resources issues/errors**: Ideally, a better performed setup will not encounter any computing resources issues/errors including but not limited to the Out-Of-Memory (OOM) error. 

## Methods
To evalute the hardware performance,  two deep learning models are deployed for benchmarking purpose. The first model is a modified VGG19 based on a study by Deitsch et al. (**Model A**) [1], and the other model is a modified concatenated model proposed in a study from Rahimzadeh et al. (**Model B**) [2]. These models were previously implemented in Vo et al [3]. The model compilation, training and validation practices are similar to those mentioned in Vo et al [3]. Besides, the mixed precision policy is applied for model training to make it run faster and consume less memory.

![](images/ModelA.png)
Figure 1: Network architecture of **Model A**. This model consists of a **VGG19 convolutional base** followed by four **convolutional layers**, a **Global Average Pooling** layer, and finally three **fully-connected neural** layers (the **Dropout** layers are excluded, but they are still presented in the actual implementation) [3].

![](images/ModelB.png)
Figure 2: Network architecture of **Model B**. This model consists of two separate **convolutional bases** that are **Xception** and **ResNet151V2**, followed by a **concatenated** layer, a **convolutional** layer, a **Global Average Pooling** layer, and finally three **fully-connected neural** layers (the **Dropout** layers are excluded, but they are still presented in the actual implementation) [3]. 

The following datasets for benchmarking are used:
* The **original MNIST dataset** developed by Yann LeCun et al [4].

![](images/mnist.png)
Figure 3: Visualization of the original MNIST dataset developed by Yann LeCun et al [4].

* The **Zalando MNIST dataset** developed by Han Xiao et al.
Figure 4: Visualization of the Zalando MNIST dataset developed by Han Xiao et al [5].

![](images/zalando.png)

On the other hand, the **total execution time** for General Matrix to Matrix Multiplication (GEMM) and recurrent neural network (RNN) can be considered as a simple alternative to model and validation. Basic operations in GEMM include Dense Matrix Multiplication, Sparse Matrix Multiplication, and Convolution. Thus, these operations exist in almost all deep neural networks today. Table 1 below outlines each basic operation to its corresponding role in a typical deep learning model.

Operation | Application | 
| ------------ | ------------- |
| Dense Matrix Multiplication | Dense Neural Layer |
| Sparse Matrix Multiplication | Dense Neural Layerr with Dropout |
| Convolution | Convolution Layer |

Table 1: Basic operations in GEMM and its corresponding application in deep learning.

## Results
To provide a solid baseline for comparison among different setups, we benchmarked our own computing resources and recorded the results. Table 2 below provides the information of our setup. Table 3 provides the results of our benchmark for the **total execution time** on MNIST and Zalando datasets, respectively. Table 4  provides the results of our benchmark for the **total execution time** on GEMM and RNN. Finally, Table 5 provides the results of our benchmark for the **total prediction time**.

Component | Description
| ------------ | ------------- |
| CPU | Intel(R) Core(TM) i7-10750H @2.60 GHz |
| GPU | NVIDIA GeForce RTX 2060 5980 MB |
| Memory | 16384 MB RAM |
| Storage | 476 GB |

Table 2: Information of the setup used as the baseline for comparison.


| Test    | Model A (s)   | Model B (s)   | FLOPS |
|---------|---------------|---------------|-------|
| MNIST   | <Placeholder> | <Placeholder> |       |
| Zalando | <Placeholder> | <Placeholder> |       | 

Table 3: Results of the benchmark on MNIST and Zalando datasets.


| Operation                    | Excution time (s) | FLOPS         |
|------------------------------|-------------------|---------------|
| Dense Matrix Multiplication  | <Placeholder>     | <Placeholder> |
| Sparse Matrix Multiplication | <Placeholder>     | <Placeholder> |
| Convolution                  | <Placeholder>     | <Placeholder> |
| Recurrent Neural Network     | <Placeholder>     | <Placeholder> |

Table 4: Results of the benchmark on GEMM and RNN

| Test    | Model A (s)   | Model B (s)   |
|---------|---------------|---------------|
| MNIST   | <Placeholder> | <Placeholder> |
| Zalando | <Placeholder> | <Placeholder> |

Table 5: Results of the benchmark on GEMM and RNN

When running the benchmark on Model B, we encoutered the following issue.
```
W tensorflow/core/common_runtime/bfc_allocator.cc:243] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.42GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
```
This issue is not an error, as mentioned in the message itself, but it is an indicator notifying that the task was too demanding that my GPU could not keep it.


## Acknowledgements
This project is the collaboration between Aalto University and RMIT Vietnam.

Contributors: 
* Vinh-Khuong Nguyen, Associated Lecturer, RMIT Vietnam
* Huynh Quang Nguyen Vo, Doctoral Candidate, Aalto University

## Appendix
1. For the theoretical backgrounds behind GEMM and recurrent network, please refer to the `Thereotical Background.ipynb` file.
2. For the installation of Python, Tensorflow, and other dependencies, please refer to the `Instruction Guide.ipynb` file.

## References
<a id="1">[1]</a> 
S. Deitsch, V. Christlein, S. Berger, C. Buerhop-Lutz, A. Maier, F. Gallwitz, and C. Riess, “Automatic classification of defective photovoltaic module cells in electroluminescence images,” Solar Energy, vol. 185, p. 455–468, 06-2019.


<a id="2">[2]</a>
M. Rahimzadeh and A. Attar, “A modified deep convolutional neural network for detecting COVID-19 and pneumonia from chest X-ray images based on the concatenation of Xception and ResNet50V2,” Informatics in MedicineUnlocked, vol. 19, p. 100360, 2020.


<a id ="3">[3]</a>
H. Vo, “Realization and Verification of Deep Learning Models for FaultDetection and Diagnosis of Photovoltaic Modules,” Master’s Thesis, Aalto University. School of Electrical Engineering, 2021.


<a id ="4">[4]</a>
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, 1998.

<a id = "5">[5]</a>
Xiao, K. Rasul, and R. Vollgraf, “A Novel Image Dataset for Benchmarking Machine Learning Algorithms,” 2017. https://github.com/zalandoresearch/fashion-mnist

<a id = "6">[6]</a>
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion,O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vander-plas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay,“Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

<a id = "7">[7]</a>
F. Chollet, “Keras,” 2015. https://github.com/fchollet/keras
