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
