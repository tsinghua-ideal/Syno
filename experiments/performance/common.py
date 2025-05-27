PRESET_WORKING_DIR = "./perf"

def get_specialized_model_name(model_name: str, batch_size: int, vanilla: bool = False) -> str:
    prefix = f"{model_name}-N={batch_size}"
    if vanilla:
        prefix += "-orig"
    return prefix

RESNET_34_LAYERS_NAMES = [
    # Same
    "conv_io64",
    "conv_io128",
    "conv_io256",
    "conv_io512",
    # Down
    "conv_i64_o128",
    "conv_i128_o256",
    "conv_i256_o512",
    # Residual
    "residual_i64_o128",
    "residual_i128_o256",
    "residual_i256_o512",
]

RESNET_34_LAYERS_MODELS = [
    f"resnet34layers/{layer}" for layer in RESNET_34_LAYERS_NAMES
]
