## 5. Additional Model - Reduce Number of Parameters to < 8K

### Target:
Reduce the number of parameters to < 8K and maintain accuracy-
1. Reduced the channel dimesnions
2. Reduced convolutions when size was at 6x6
3. Removed dropouts after 2 convolution layers as it was underfitting

### Results:
```
Parameters: 7,614 parameters
Best Train Accuracy: 99.05%
Best Test Accuracy: 99.46%
```

### Analysis:
>`1. Reducing dropouts helped with overcoming underfitting`
>
>`2. Last 4 epochs have an average accuracy of 99.40%.`
