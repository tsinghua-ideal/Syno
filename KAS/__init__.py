# Here, we need to load the runtime symbols for dynamic loading of kernels.
import sys
import os
# We should first import torch before loading the runtime, because the runtime may depend on torch.
import torch
import torch.nn as nn

# Initialize the Triton runtime symbols, by compiling a dummy kernel before importing KAS cpp bindings.
class _initial_compile(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
assert torch.cuda.is_available(), 'CUDA is not available'
_initial_kernel = _initial_compile().cuda()
_initial_compiled_kernel = torch.compile(_initial_kernel, backend='inductor', mode='reduce-overhead', dynamic=False, fullgraph=False)
_initial_compiled_kernel(torch.ones([1, 3, 32, 32]).cuda())
del _initial_kernel

# Now import the bindings. First backup the flags.
old_flags = sys.getdlopenflags()
# But we need to resolve symbols for the main program first, because otherwise the symbols may be mistakenly replaced by the runtime.
sys.setdlopenflags(os.RTLD_LOCAL | os.RTLD_NOW)
import kas_cpp_bindings
# Now we can safely load the runtime, adding the symbols to the global scope.
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
import kas_runtime
# Restore the original flags.
sys.setdlopenflags(old_flags)
del old_flags

from .Assembler import Assembled, Assembler
from .Explorer import Explorer
from .KernelPack import KernelPack, KernelLoader
from .Node import Next, Path, Arc, Node, MockNodeMetadata, MockNode, MockVisitedNode
from .Placeholder import Placeholder
from .Sampler import Sampler, CodeGenOptions, MockSampler
from .Statistics import Statistics
from .Utils import NextSerializer, init_weights
