
CUDA_VISIBLE_DEVICES=0 python test_run.py --model torchvision/resnet18 --dataset imagenet --kas-use-orig-model --imagenet-log-folder logs/resnet18_orig_eval --num-classes 1000 --batch-size 1024 --compile --ckpt-path results/quantization/resnet18_orig.pt --quant > logs/imagenet/imagenet_orig_eval_quant.log 2>&1

CUDA_VISIBLE_DEVICES=0 python test_run.py --model torchvision/resnet18 --dataset imagenet --kas-test-path results/manual_kernels/kernel_07889 --imagenet-log-folder logs/resnet18_ours_eval --num-classes 1000 --batch-size 1024 --compile --ckpt-path results/quantization/resnet18_ours.pt --quant > logs/imagenet/imagenet_ours_eval_quant.log 2>&1

CUDA_VISIBLE_DEVICES=0 python test_run.py --model torchvision/resnet18 --dataset imagenet --kas-test-path results/manual_kernels/Conv2d_Conv1d --imagenet-log-folder logs/resnet18_tiledconv_eval --batch-size 1024 --num-classes 1000 --compile --ckpt-path results/quantization/resnet18_tiledconv.pt --quant > logs/imagenet/imagenet_tiledconv_eval_quant.log 2>&1

CUDA_VISIBLE_DEVICES=0 python test_run.py --model torchvision/resnet18 --dataset imagenet --kas-use-orig-model --imagenet-log-folder logs/resnet18_orig --batch-size 1024 --num-classes 1000 --compile > logs/imagenet/imagenet_orig_compile.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python test_run.py --model torchvision/resnet18 --dataset imagenet --kas-test-path results/manual_kernels/kernel_07889 --imagenet-log-folder logs/resnet18_ours --batch-size 1024 --num-classes 1000 --compile > logs/imagenet/imagenet_ours.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python test_run.py --model torchvision/resnet18 --dataset imagenet --kas-test-path results/manual_kernels/Conv2d_Conv1d --imagenet-log-folder logs/resnet18_tiledconv --batch-size 1024 --num-classes 1000 --compile > logs/imagenet/imagenet_tiledconv.log 2>&1 &