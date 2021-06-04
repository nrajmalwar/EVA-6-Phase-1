# MNIST Classification using Target, Results and Analysis Strategy

The objective is to train the model in 4 steps building a foundation and subsequently improving on it with each step. We want to achieve higher than 99.40% accuracy is less than 10,000 parameters and within 15 epochs.

The model is trained in 4 steps-

## 1. Skeleton Model
#### Target
  1. Build the model with right setup of transforms, dataloader, training and testing functions and loops.
  2. Build the skeleton of the model with less than 10,000 parameters following output size and receptive field calculations.
  3. Place the MaxPooling (RF = 5) and GlobalAveragePooling (last but one layer) at appropriate positions
  4. Print the misclassified images to analyse.

#### Result
    Parameters - 9,264
    Best Train Acc - 98.82%
    Best Test Acc - 98.76%

#### Analysis
>`1. Model is light and very slightly overfitting.`
>
>`2. Model has potential to attain more train accuracy and in turn higher test accuracy.`
>
>`3. Train accuracy is not increasing in a stable and gradual manner.`
>
>`4. We can expect model stabilization through regularization and an increase in accuracy with BatchNormalization.`
>
>`5. Misclassifed images shows that model makes basic mistakes between pairs (4,9), (5,2), (8,9). We will look to improve on this later. The general classification seems to be good for a skeleton model.`

## 2. Regularization and BatchNormalization

#### Target

1. Add dropout value of 0.1 to all the layers except the last one
2. Add BatchNormalization to all the layer except the last one

#### Result
    Parameters - 9,436 (slight increase due to BatchNormalization)
    Best Train Acc - 99.20% (Increase)
    Best Test Acc - 99.42% (Increase)

#### Analysis
> ``` 1. Regularization and BatchNormalization has resulted in stabilization of train accuracy and increase in test accuracy.```
> 
> ``` 2. Model is slightly underfitting, which means there is scope to increase the accuracy further. ```
> 
> ``` 3. Test accuracy iof 99.42% is achieved only in the last epoch.```
> 
> ``` 4. Next we apply data augmentation to train the model harder and achieve 99.42% test accuracy earlier. Analysing the misclassified shows us that some images require rotation for the model to understand the visual differences between some numbers like (7,1), (2,1) and (4,9). ```
> 
> ``` 5. We will also apply slight distortion to train model to rectify images which are harder to read otherwise. Distortion can help improve readibility of images that are difficult even for the human eye.```

## 3. Image Augmentations

### Target
1. Add rotation of (-6.0, 6.0) degrees to the dataset. Fill the void with pixel values of 33 (mean \* max pixel value = 0.1307 * 255)
2. Add slight distortion of scale 0.2 and probability of 50%.

### Result
```
Parameters - 9,436 (No change)
Best Train Acc - 98.97% (slight decrease due to augmentation)
Best Test Acc - 99.41% (No change)
```
### Analysis

>`1. Augmentation has slightly decreased the train accuracy as the training has become harder.`
>
>`2. Test accuracy of 99.41% is achieved at the 7th epoch.`
>
>`3. Test accuracy is not stable at 99.41% due to high learning rate.`
>
>`4. We will use an LR Scheduler to achieve 99.40%+ accuracy faster and consistently.`
>
>`5. We will also change the batch size from 128 to 256 to see if model trains faster.`

## 4. LR Scheduler

### Target
1. Apply LR Scheduler using Lambda LR and a multiplicative factor of 0.85 ** epoch

### Result
```
Parameters - 9,436 (No change)
Best Train Acc - 99.05% (No change)
Best Test Acc - 99.53% (Increase)
```
### Analysis 
>`1. With LR Scheduler, target accuracy of 99.40%+ is achieved at 7th epoch and stays consistently.`
>
>`2. Due to smaller learning rate in later epochs, accuracy bumps up to 99.53%.`
>
>`3. Last 3 epochs have an average accuracy of 99.50%.`
>
>`4. Changing batch size to 256 decreases the training time per epoch slightly by 1 second.`

