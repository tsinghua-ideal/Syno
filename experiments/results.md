### Results Aggregation

| Kernel | Accuracy (resnet18, cifar100) | Accuracy (resnet34, cifar100) | Accuracy (resnet18, ImageNet) | FLOPs (resnet18) | Performance (resnet18) | Performance (resnet34) |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Baseline | 0.7883 | 0.7945 | 0.7084 | 1.824G | 38.8ms | 71.4ms |
| Conv2d | 0.7831 | 0.7922 |  | 1.825G | 35.7ms | |
| Group Conv OAS | 0.7660 | 0.7692 |  | 0.2778G | 8.48ms | 12.5ms |
| Kernel_07923 | 0.7816 | 0.7948 | 0.6975 | 1.266G | 16.5ms | 29.6ms |
