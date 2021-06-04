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
