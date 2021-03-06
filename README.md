# Deep-Learning-Hardware-Benchmark
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4905229.svg)](https://doi.org/10.5281/zenodo.4905229)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4905213.svg)](https://doi.org/10.5281/zenodo.4905213)

## Introduction
This repository contains our proposed implementation for benchmarking to evaluate whether a setup of hardware is feasible for complex deep learning projects.

## Scope
* The benchmark evaluates the performance of a setup having a single CPU, a single GPU, RAM and memory storage. The performance of multi-CPUs/multi-GPUs or server-based is included in our scope.
* The benchmark is built on the **Anaconda** distribution of Python, and the **Jupyter Notebook** computational environment. The deep learning models mentioned in this benchmarked are implemented using the **Keras** application programming interface (API).
* Our goal is to develop a verified approach to conduct the hardware benchmark that is quick and easy to use. 

## Evaluation metrics
There are various metrics to benchmark the performance capabilities of a setup for deep learning purposes. Here, the following metrics are used:
1. **Total execution time**: the **total execution time** includes both the **total training time** and the **total validation time** of a deep learning model on a dataset after a defined number of epochs. Here, the number of epochs is 100. The lower the **total execution time** the better.
2. **Total inference time**: the **total inference time** includes both the **model loading time** (the time required to fully load a set of pre-trained weights to implement a model) and the **total prediction time** of a deep learning model on a test dataset. Similar to the **total execution time**, the lower the **total inference time** the better.
3. **FLOPS**: the performance capability of a CPU or GPU can be measured by counting the number of floating operation points (FLO) it can execute per second. Thus, the higher the **FLOPS**, the better. 
4. **Computing resources issues/errors**: Ideally, a better-performed setup will not encounter any computing resources issues/errors including but not limited to the Out-Of-Memory (OOM) error.
5. **Bottlenecking**: to put it simply, bottlenecking is a subpar performance that is caused by the inability of one component to keep up with the others, thus slowing down the overall ability of a setup to process data. Here, our primary concern is the bottlenecking between CPU and GPU. The **bottlenecking factor** is measured using an online tool: [Bottleneck Calculator](https://pc-builds.com/calculator/)

## Methods
1. To evaluate the hardware performance, two deep learning models are deployed for benchmarking purpose. The first model is a modified VGG19 based on a study by Deitsch et al. (**Model A**) [1], and the other model is a modified concatenated model proposed in a study from Rahimzadeh et al. (**Model B**) [2]. These models were previously implemented in Vo et al [3]. The model compilation, training and validation practices are similar to those mentioned in Vo et al [3]. Besides, several optimization practices such as mixed precision policy are applied for model training to make it run faster and consume less memory.

![](images/ModelA.png)
Figure 1: Network architecture of **Model A**. This model consists of a **VGG19 convolutional base** followed by four **convolutional layers**, a **Global Average Pooling** layer, and finally three **fully-connected neural** layers (the **Dropout** layers are excluded, but they are still presented in the actual implementation) [3].

![](images/ModelB.png)
Figure 2: Network architecture of **Model B**. This model consists of two separate **convolutional bases** that are **Xception** and **ResNet151V2**, followed by a **concatenated** layer, a **convolutional** layer, a **Global Average Pooling** layer, and finally three **fully-connected neural** layers (the **Dropout** layers are excluded, but they are still presented in the actual implementation) [3]. 

We used the following datasets for benchmarking: the **original MNIST dataset** by LeCun et al., and the **Zalando MNIST dataset** by Xiao et al.

2. On the other hand, we also proposed another approach for benchmarking that is much simpler and quicker: evaluating the **total execution time** for a combination of basic operations. These basic operations include General Matrix to Matrix Multiplication (GEMM), 2D-Convolution (Convolve2D) and Recurrent Neural Network (RNN), and exist in almost all deep neural networks today [4]. Table 1 below outlines how these basic operations are applicable in Deep Learning.

Test | Application | 
| ------------ | ------------- |
| Dense Matrix Multiplication (DMM)| Dense Neural Network |
| Sparse Matrix Multiplication (SMM)| Dense Neural Network with Dropout |
| Stacking 2D-Convolution (Convolve2D) | Convolution Neural Network|
| Recurrent Neural Network (RNN) | Dealing with Time Series Data/ Natural Language Processing |

Table 1: Application of basic operations in Deep Learning. 

We implemented our alternative approach based on the DeepBench work by Baidu [5]:
* In DMM, we defined a matrix C as a product of `(MxN)` and `(NxK)` matrices. For example, `(3072,128,1024)` means the resulting matrix is a product of `(3072x128)` and `(128x1024)` matrices. To benchmark, we implemented five different multiplications, and measured the overall **total execution time** of these five. These multiplications included `(3072,128,1024)`, `(5124,9124,2560)`, `(2560,64,2560)`, `(7860,64,2560)`, and `(1760,128,1760)`.
* In SMM, we defined a matrix C as a product of `(MxN)` and `(NxK)` matrices, and `(100 - Dx100)%` of the `(MxN)` matrix is obmitted. For instance, `(10752,1,3584,0.9)` means the resulting matrix is a product of `(10752x1)` and `(1x3584)` matrices, while 10% of the `(10752x1)` matrix is obmitted. To benchmark, we implemented four different multiplications, and measured the overall **total execution time** of these five. These multiplications included `(10752,1,3584,0.9)`, `(7680,1500,2560,0.95)`, `(7680,2,2560,0.95)`, and `(7680,1,2560,0.95)`.
* In Convolve2D, we defined a simple model containing only convolution layers and pooling layers as in Figure 5, and measured the resulting **total execution time**. The dataset used for this training this model is the **Zalando MNIST** by Xiao et al.
* We did not implement the **RNN** due to several issues caused by the new version of Keras.

![](images/Conv1.png)

Figure 3: A simple model containing only convolution layers and pooling layers for the alternative benchmark approach.

3. To evaluate **total inference time**, we loaded the already trained weights from our models (denoted as **Model A-benchmarked** and **Model B-benchmarked**, respectively) which has the best validation accuracy, and conducted a prediction run on the test set from the **Zalando MNIST**. These files are available on Zenodo: [Inference Models](https://zenodo.org/record/4905213#.YL1-P_kzaUk)

## Results
To provide a solid baseline for comparison among different setups, we benchmarked our computing resources and recorded the results. Table 2 below provides the information on our setup. 

|         |           Setup 1           |           Setup 2           |           Setup 3           |           Setup 4           |
|---------|:---------------------------:|:---------------------------:|:---------------------------:|:---------------------------:|
| CPU     | Core(TM) i7-10750H @2.60GHz | Core(TM) i9-10900K @3.70GHz | Core(TM) i7-8700K @3.70GHz  | Core(TM) i9-7920X @2.90GHz  |
| GPU     | GeForce RTX 2060 5980MB     | GeForce RTX 3070 8031MB     | GeForce RTX 2070 8031MB     | GeForce RTX 2070 8031MB     |
| Memory  | 16384MB RAM @ DDR4 3200MHz  | 32384MB RAM @ DDR4 3200MHz  | 32768MB RAM @ DDR4 3200MHz  | 131072MB RAM @ DDR4 3200MHz |
| Storage | 476GB SSD                   | 968GB SSD                   | 468GB SSD                   | 468GB SSD                   |

Table 2: Information of the setups used as the benchmark baseline for comparison.

Below is the summarized visualizations of our baseline results (the results for the alternative approach are obmited) that are created using SAS JMP. We included a compiled results in `results/results.csv` for reference.

![](images/Total%20Execution%20Time.png)
Figure 4: Total Execution Time between the setups when trained on the **original MNIST** and **Zalando** datasets. 

![](images/Total%20Inference%20Time.png)
Figure 5: Total Inference Time between the setups when validated on the **Zalando** datasets. 

![](images/Bottleneck%26FLOPS.png)
Figure 6: The bottlenecking factor and the number of theoretical FLOPS between the setups.

Additionally, the Machine Learning Benchmark Organization (MLPerf) also provided a comprehensive list of training and testing results from many different setups for comparison. The latest results (v.07 as of May 2021) are available at: https://mlcommons.org/en/

## Acknowledgements

Contributors: 
* Vinh-Khuong Nguyen (Dr), Associated Lecturer, RMIT Vietnam.
* Huynh Quang Nguyen Vo (MSc), Aalto University.

## Appendix
1. For the installation of Python, Tensorflow, and other dependencies, please refer to the `Instruction Guide (Python 3.7).ipynb` and `Instruction Guide (Python 3.8).ipynb` files. So far, all programs are developed under Python 3.7.

2. Visualizations of the respective datasets used in this repository.
  
![](images/mnist.png)
Figure 7: Visualization of the original MNIST dataset developed by Yann LeCun et al [4].

![](images/zalando.png)
Figure 8: Visualization of the Zalando MNIST dataset developed by Han Xiao et al [5].

3. List of the packages used in our implementation is provided in the file `packages.txt` for reproduction.

## References
<a id="1">[1]</a> S. Deitsch, V. Christlein, S. Berger, C. Buerhop-Lutz, A. Maier, F. Gallwitz, and C. Riess, ???Automatic classification of defective photovoltaic module cells in electroluminescence images,??? Solar Energy, vol. 185, p. 455???468, 06-2019.


<a id="2">[2]</a> M. Rahimzadeh and A. Attar, ???A modified deep convolutional neural network for detecting COVID-19 and pneumonia from chest X-ray images based on the concatenation of Xception and ResNet50V2,??? Informatics in MedicineUnlocked, vol. 19, p. 100360, 2020.


<a id ="3">[3]</a> H. Vo, ???Realization and Verification of Deep Learning Models for FaultDetection and Diagnosis of Photovoltaic Modules,??? Master???s Thesis, Aalto University. School of Electrical Engineering, 2021.


<a id ="4">[4]</a> P. Warden, "Why GEMM is at the heart of deep learning," Pete Warden's Blog, 2015. Available at: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

<a id = "5">[5]</a> Baidu Research, "Benchmarking Deep Learning operations on different hardware". Available at: https://github.com/baidu-research/DeepBench

<a id = "6">[6]</a> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, 1998.

<a id = "7">[7]</a> Xiao, K. Rasul, and R. Vollgraf, ???A Novel Image Dataset for Benchmarking Machine Learning Algorithms,??? 2017. https://github.com/zalandoresearch/fashion-mnist

<a id = "8">[8]</a> F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion,O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vander-plas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay,???Scikit-learn: Machine learning in Python,??? Journal of Machine Learning Research, vol. 12, pp. 2825???2830, 2011.

<a id = "9">[9]</a> F. Chollet, ???Keras,??? 2015. Available at: https://github.com/fchollet/keras
  
<a id = "10">[10]</a> ML Commons. Available at: https://mlcommons.org/en/

<a id = "11"></a>[11] S.  Poppi,  M.  Cornia,  L.  Baraldi,  and  R.  Cucchiara,  ???Revisiting the evaluation of class activation mapping for explainability: A novel metric and experimental analysis,??? 2021.
  
<a id = "12"></a>[12] W.  Dai and  D.   Berleant,  ???Benchmarking contemporary deep learning hardware and frameworks:  A  survey of qualitative metrics,??? 2019 IEEE First International Conference on Cognitive Machine Intelligence (CogMI), Dec 2019.
