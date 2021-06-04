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
