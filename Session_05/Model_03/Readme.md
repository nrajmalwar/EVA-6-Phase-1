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
