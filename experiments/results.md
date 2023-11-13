### Results Aggregation

| Kernel | Accuracy (resnet18, cifar100) | Accuracy (resnet34, cifar100) | Accuracy (resnet18, ImageNet) | FLOPs (resnet18) |
| ------| ------| ------| ------| ------| 
| Baseline (Conv2d) | 0.7831 | 0.7922 | 0.7071 | 1.825G |
| Group Conv OAS |  |  |  | 0.2778G |
| Kernel_07923 | 0.7816 | 0.7948 |  | 1.266G |
