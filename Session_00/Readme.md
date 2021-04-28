
1. What are Channels and Kernels (according to EVA)?

2. Why should we (nearly) always use 3x3 kernels?
---
3. How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
```
    1.      199X199 | 3X3 > 197X197
    2.      199X199 | 3X3 > 197X197
    3.      199X199 | 3X3 > 197X197
    4.      199X199 | 3X3 > 197X197
    5.      199X199 | 3X3 > 197X197
    6.      199X199 | 3X3 > 197X197
    7.      199X199 | 3X3 > 197X197
    8.      199X199 | 3X3 > 197X197
    9.      199X199 | 3X3 > 197X197
    10.     199X199 | 3X3 > 197X197
    11.     199X199 | 3X3 > 197X197
 ```
 ---
5. How are kernels initialized? 
6. What happens during the training of a DNN?
