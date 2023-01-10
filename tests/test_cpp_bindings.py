import kas_cpp_bindings

def test_sample():
    options = kas_cpp_bindings.SampleOptions()
    sampler = kas_cpp_bindings.Sampler("[H,W]", "[N,C,H,W]", options)
    assert sampler.isFinal([0])
    kernel = sampler.realize([0])
    cg_opt = kas_cpp_bindings.CodeGenOptions(False, kas_cpp_bindings.CodeGenOptions.AutoScheduler.Li2018)
    kernel.generate("build/kernel_0", "kernel_0", cg_opt)
