import kas_cpp_bindings
from kas_cpp_bindings import SampleOptions

class Sampler():
    def __init__(self, input_shape: str, output_shape: str, options: SampleOptions):
        self._sampler = kas_cpp_bindings.Sampler(input_shape, output_shape, options)
