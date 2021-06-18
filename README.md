# EVA-6-Phase-1
Assignment submissions from the EVA-6 Phase 1 program. EVA is an exhaustive and updated Deep Vision program. Phase 1 consists of fundamentals of deep learning and concludes with Transformers and Attention Mechanism for images.

## [Session 0: Background and Basics - Machine Learning intuition](Session_00)

Answer questions on the basics of convolutional neural network

## [Session 3: Pytorch](Session_03)

Write a neural network that takes two inputs
* an image from MNIST dataset
* a random number between 0 and 9

and gives the output
* the "number" that was represented by the MNIST image
* the "sum" of this number with the random number that was generated and sent as the input to the network

## [Session 4: Backpropagation and Architectural Basics](Session_04)

* [Train a neural network on excel sheet with all the backpropagation calculations involved](Session_04/Backpropagation_Calculations)
* [Train MNIST Classifier Network with less than 20k parameters and 99.4% validation accuracy](Session_04/Architectural_Basics)

## [Session 5: Coding Drill Down](Session_05)
* Train a neural network using the **Target, Results and Analysis** strategy to achieve 99.50% test accuracy on MNIST dataset in less than 10,000 parameters and 15 epochs. The model is trained in 4 steps-

     [1. Skeleton Model](Session_05/Model_01)

     [2. Regularization and BatchNormalization](Session_05/Model_02)

     [3. Image Augmentations](Session_05/Model_03)

     [4. LR Scheduler](Session_05/Model_04)
     
     [5. Additional Model < 8K Parameters](Model_05_Additional_Model)
     
## [Session 6: Batch Normalization and Regularization ](Session_06)
* Train 3 neural networks using the following normalization techniques-
     1. Batch Normalization + L1 Regularization
     2. Layer Normalization
     3. Group Nomalization

## [Session 7: Advanced Concepts](Session_07)
* Train a neural network on the CIFAR10 dataset with the following features-
     1. Use albumentations library and apply:
         - Horizontal Flip
         - ShiftScaleRotate
         - CoarseDropout
     2. Achieve 85% test accuracy
     3. Less than 200k parameters
     4. one of the layers must use Depthwise Separable Convolution
     5. one of the layers must use Dilated Convolution
