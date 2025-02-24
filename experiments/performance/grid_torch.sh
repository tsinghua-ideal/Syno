function tune() {
    local model=$1
    local result_dir=$2

    echo "Tuning model: $model, result_dir: $result_dir"
    if [ -z "$result_dir" ]; then
        # Vanilla model, no need to specify result_dir
        python perf_torch.py --device cuda --mode max-autotune --model $model
    else
        python perf_torch.py --device cuda --mode max-autotune --model $model --result-dir $result_dir
    fi
}

# # ResNet-18 cannot share codegen with ResNe-34

# rsync -av ./results/resnet18-kernels/* ./results/resnet-good-kernels/

# tune torchvision/resnet18
# for par_dir in ./results/resnet-good-kernels/*; do
#     for dir in $par_dir/*; do
#         tune torchvision/resnet18 $dir
#     done
# done

rsync -av ./results/resnet34-kernels/* ./results/resnet-good-kernels/

tune torchvision/resnet34
for par_dir in ./results/resnet-good-kernels/*; do
    for dir in $par_dir/*; do
        tune torchvision/resnet34 $dir
    done
done

for layer in "conv_io64" "conv_io128" "conv_io256" "conv_io512" "conv_i64_o128" "conv_i128_o256" "conv_i256_o512" "residual_i64_o128" "residual_i128_o256" "residual_i256_o512"; do
    tune resnet34layers/$layer
    tune resnet34layers/$layer ./results/resnet-good-kernels/0.6x/07889_15252107013978896537
    tune resnet34layers/$layer ./results/resnet-good-kernels/0.2x/07754_18091915762600937904
    for seq in ./results/nas-pte/*; do
        tune resnet34layers/$layer $seq
    done
done

tune torchvision/resnext29_2x64d
for dir in ./results/resnext-good-kernels/*; do
    tune torchvision/resnext29_2x64d $dir
done

tune torchvision/efficientnet_v2_s
for dir in ./results/efficientnet-good-kernels/*; do
    tune torchvision/efficientnet_v2_s $dir
done

tune torchvision/densenet121
for dir in ./results/densenet-good-kernels/*; do
    tune torchvision/densenet121 $dir
done
