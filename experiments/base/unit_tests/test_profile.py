import os, sys, shutil
from types import SimpleNamespace

if os.getcwd() not in sys.path: 
    sys.path.append(os.getcwd())
from models import get_model


def get_model_args(model_name: str, batch_size=2, kas_replace_placeholder=None) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_name, 
        batch_size=batch_size, 
        kas_replace_placeholder=kas_replace_placeholder,
        seed=None,
        kas_depth=6,
        kas_min_dim=1,
        kas_max_dim=8,
        kas_max_tensors=3,
        kas_max_reductions=4,
        kas_max_flops_ratio=2,
        kas_sampler_workers=1,
        compile=True,
        kas_scheduler_cache_dir='.scheduler-cache',
    )

def test_fc_profile(batch_size=2):
    orig_model_args = get_model_args('FCNet', batch_size)
    manual_model_args = get_model_args('FCNet', batch_size, 'linear')
    
    model_orig = get_model(orig_model_args)
    model_manual = get_model(manual_model_args)
    
    flops_orig, params_orig = model_orig.profile(batch_size)
    flops_manual, params_manual = model_manual.profile(batch_size)
    
    flops_gt = 28 * 28 * 64 + 0 + 4 * 64 + 64 * 10
    params_gt = 28 * 28 * 64 + 64 * 2 + 64 * 10 + 10
    
    assert flops_orig == flops_manual == flops_gt, f'FLOPs mismatch: {flops_orig} vs {flops_manual} vs {flops_gt}'
    assert params_orig == params_manual == params_gt, f'Params mismatch: {params_orig} vs {params_manual} vs {params_gt}'
    
    print("PASSED: test_fc_profile")
    
def test_conv_profile(batch_size=2):
    orig_model_args = get_model_args('ConvNet', batch_size)
    manual_model_args = get_model_args('ConvNet', batch_size, 'Conv')
    
    model_orig = get_model(orig_model_args)
    model_manual = get_model(manual_model_args)
    
    flops_orig, params_orig = model_orig.profile(batch_size)
    flops_manual, params_manual = model_manual.profile(batch_size)
    
    # Conv: C_out * H_out * W_out * C_in * H * W
    # Ignore bias in FLOPs
    flops_gt = \
        64 * 32 * 32 * 3 * 3 * 3 + 4 * 64 * 32 * 32 + 0 + \
        96 * 16 * 16 * 64 * 3 * 3 + 4 * 96 * 16 * 16 + 0 + \
        128 * 8 * 8 * 96 * 3 * 3 + 4 * 128 * 8 * 8 + 0 + \
        4 * 4 * 128 * 512 + 0 + \
        512 * 256 + 0 + \
        256 * 10
        
    # Conv: C_in * C_out * H * W
    params_gt = \
        3 * 64 * 3 * 3 + 2 * 64 + 0 +\
        64 * 96 * 3 * 3 + 2 * 96 + 0 +\
        96 * 128 * 3 * 3 + 2 * 128 + 0 + \
        4 * 4 * 128 * 512 + 512 + 0 + \
        512 * 256 + 256 + 0 + \
        256 * 10 + 10
    
    assert flops_orig == flops_manual == flops_gt, f'FLOPs mismatch: {flops_orig} vs {flops_manual} vs {flops_gt}'
    assert params_orig == params_manual == params_gt, f'Params mismatch: {params_orig} vs {params_manual} vs {params_gt}'
    
    print("PASSED: test_conv_profile")


if __name__ == '__main__':
    test_fc_profile()
    test_conv_profile()
    shutil.rmtree('.scheduler-cache') # remove the cache
