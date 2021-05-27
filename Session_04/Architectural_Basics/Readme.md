# MNIST Classification using stripped down network with less than 20,000 parameters
The objective is to optimize a network combining many deep learning techniques on the layers and use very few parameters to achieve high accuracy

## Data Preprocessing

* Create dataloader object for the train and test data for the MNIST dataset and apply normalization using mean and std deviation values of (0.1307,), (0.3081,)
* Batch size = 28

## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 20, 24, 24]           2,900
              ReLU-6           [-1, 20, 24, 24]               0
       BatchNorm2d-7           [-1, 20, 24, 24]              40
           Dropout-8           [-1, 20, 24, 24]               0
            Conv2d-9           [-1, 24, 22, 22]           4,344
             ReLU-10           [-1, 24, 22, 22]               0
      BatchNorm2d-11           [-1, 24, 22, 22]              48
        MaxPool2d-12           [-1, 24, 11, 11]               0
           Conv2d-13           [-1, 10, 11, 11]             250
             ReLU-14           [-1, 10, 11, 11]               0
      BatchNorm2d-15           [-1, 10, 11, 11]              20
           Conv2d-16             [-1, 16, 9, 9]           1,456
             ReLU-17             [-1, 16, 9, 9]               0
      BatchNorm2d-18             [-1, 16, 9, 9]              32
          Dropout-19             [-1, 16, 9, 9]               0
           Conv2d-20             [-1, 20, 7, 7]           2,900
             ReLU-21             [-1, 20, 7, 7]               0
      BatchNorm2d-22             [-1, 20, 7, 7]              40
          Dropout-23             [-1, 20, 7, 7]               0
           Conv2d-24             [-1, 24, 5, 5]           4,344
             ReLU-25             [-1, 24, 5, 5]               0
      BatchNorm2d-26             [-1, 24, 5, 5]              48
          Dropout-27             [-1, 24, 5, 5]               0
           Conv2d-28             [-1, 16, 5, 5]             400
             ReLU-29             [-1, 16, 5, 5]               0
      BatchNorm2d-30             [-1, 16, 5, 5]              32
           Linear-31                   [-1, 10]             170
================================================================
Total params: 17,216
Trainable params: 17,216
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.09
Params size (MB): 0.07
Estimated Total Size (MB): 1.16
----------------------------------------------------------------
```

## Model Training and Evaluation
```
Epoch number: 1
loss=2.466463327407837 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.29it/s]
Test set: Average loss: 2.6932, Accuracy: 9862/10000 (99%)
Epoch number: 2
loss=2.430588960647583 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.64it/s]
Test set: Average loss: 2.6594, Accuracy: 9875/10000 (99%)
Epoch number: 3
loss=2.394151210784912 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.90it/s]
Test set: Average loss: 2.6450, Accuracy: 9901/10000 (99%)
Epoch number: 4
loss=2.35239315032959 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.61it/s]
Test set: Average loss: 2.6340, Accuracy: 9910/10000 (99%)
Epoch number: 5
loss=2.377195358276367 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.61it/s]
Test set: Average loss: 2.6280, Accuracy: 9918/10000 (99%)
Epoch number: 6
loss=2.375176191329956 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.08it/s]
Test set: Average loss: 2.6232, Accuracy: 9926/10000 (99%)
Epoch number: 7
loss=2.3512346744537354 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.96it/s]
Test set: Average loss: 2.6201, Accuracy: 9924/10000 (99%)
Epoch number: 8
loss=2.3895585536956787 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.47it/s]
Test set: Average loss: 2.6196, Accuracy: 9935/10000 (99%)
Epoch number: 9
loss=2.3422505855560303 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.45it/s]
Test set: Average loss: 2.6141, Accuracy: 9931/10000 (99%)
Epoch number: 10
loss=2.3922512531280518 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.59it/s]
Test set: Average loss: 2.6101, Accuracy: 9935/10000 (99%)
Epoch number: 11
loss=2.4024271965026855 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.75it/s]
Test set: Average loss: 2.6145, Accuracy: 9935/10000 (99%)
Epoch number: 12
loss=2.3560290336608887 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.95it/s]
Test set: Average loss: 2.6122, Accuracy: 9934/10000 (99%)
Epoch number: 13
loss=2.4060633182525635 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.64it/s]
Test set: Average loss: 2.6118, Accuracy: 9944/10000 (99%)
Epoch number: 14
loss=2.3257551193237305 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.21it/s]
Test set: Average loss: 2.6095, Accuracy: 9939/10000 (99%)
Epoch number: 15
loss=2.3567306995391846 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.66it/s]
Test set: Average loss: 2.6059, Accuracy: 9946/10000 (99%)
Epoch number: 16
loss=2.37717342376709 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 36.06it/s]
Test set: Average loss: 2.6087, Accuracy: 9943/10000 (99%)
Epoch number: 17
loss=2.358330488204956 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.84it/s]
Test set: Average loss: 2.6126, Accuracy: 9942/10000 (99%)
Epoch number: 18
loss=2.3288562297821045 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.65it/s]
Test set: Average loss: 2.6104, Accuracy: 9945/10000 (99%)
Epoch number: 19
loss=2.3160338401794434 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.84it/s]
Test set: Average loss: 2.6098, Accuracy: 9941/10000 (99%)
Epoch number: 20
loss=2.38149356842041 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 36.07it/s]
Test set: Average loss: 2.6136, Accuracy: 9941/10000 (99%)
```

## Observations
