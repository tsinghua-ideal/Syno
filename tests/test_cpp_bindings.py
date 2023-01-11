from kas_cpp_bindings import *

def test_sample():
    options = SampleOptions()
    sampler = Sampler("[H,W]", "[N,C,H,W]", options)
    assert sampler.isFinal([0])
    kernel = sampler.realize([0])
    cg_opt = CodeGenOptions(False, CodeGenOptions.AutoScheduler.Li2018)
    kernel.generate("build/kernel_0", "kernel_0", cg_opt)
