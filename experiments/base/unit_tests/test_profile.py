import os, sys
from types import SimpleNamespace

if os.getcwd() not in sys.path: 
    sys.path.append(os.getcwd())
from models import get_model


def get_model_args(model_name: str, batch_size=1, kas_replace_placeholder=None) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_name, 
        batch_size=batch_size, 
        kas_replace_placeholder=kas_replace_placeholder,
        seed=None,
        kas_depth=3,
        kas_min_dim=1,
        kas_max_dim=8,
        kas_max_tensors=3,
        kas_max_reductions=4,
        kas_max_flops=1e9,
        kas_scheduler_cache_dir='.scheduler-cache',
    )

def test_fc_profile(batch_size=1):
    orig_model_args = get_model_args('FCNet', batch_size)
    manual_model_args = get_model_args('FCNet', batch_size, 'linear')
    
    model_orig = get_model(orig_model_args)
    # FIXME: manual_model_args will cause SEGFAULT 
    model_manual = get_model(manual_model_args)
    
    flops_orig, params_orig = model_orig.profile(batch_size)
    flops_manual, params_manual = model_manual.profile(batch_size)
    
    assert flops_orig == flops_manual, f'FLOPs mismatch: {flops_orig} vs {flops_manual}'
    assert params_orig == params_manual, f'Params mismatch: {params_orig} vs {params_manual}'
    
    print("PASSED: test_fc_profile")
    
def test_conv_profile(batch_size=1):
    model_orig = get_model(get_model_args('ConvNet', batch_size))
    model_manual = get_model(get_model_args('ConvNet', batch_size, 'Conv'))
    
    flops_orig, params_orig = model_orig.profile(batch_size)
    flops_manual, params_manual = model_manual.profile(batch_size)
    
    assert flops_orig == flops_manual, f'FLOPs mismatch: {flops_orig} vs {flops_manual}'
    assert params_orig == params_manual, f'Params mismatch: {params_orig} vs {params_manual}'
    
    print("PASSED: test_conv_profile")


if __name__ == '__main__':
    test_fc_profile(4)
    test_conv_profile(4)
