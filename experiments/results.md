### Results Aggregation

| Kernel | Accuracy (resnet18, cifar100) | Accuracy (resnet34, cifar100) | Accuracy (resnet18, ImageNet) | FLOPs (resnet18) | Performance (resnet18) |
| ------| ------| ------| ------| ------| ------ |
| Baseline | 0.7883 | 0.7945 |  | 1.824G | 38.8ms |
| Conv2d | 0.7831 | 0.7922 | 0.7071 | 1.825G | 35.7ms |
| Group Conv OAS |  |  |  | 0.2778G | 8.48ms |
| Kernel_07923 | 0.7816 | 0.7948 |  | 1.266G | 16.5ms |
