PRESET_WORKING_DIR = "./perf"

def get_specialized_model_name(model_name: str, batch_size: int, vanilla: bool = False) -> str:
    prefix = f"{model_name}-N={batch_size}"
    if vanilla:
        prefix += "-orig"
    return prefix
