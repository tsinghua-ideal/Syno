def get_specialized_model_name(model_name: str, batch_size: int) -> str:
    return f"{model_name}-N={batch_size}"
