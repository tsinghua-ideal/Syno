import importlib
import os
from typing import Tuple
import sys

import tvm
from tvm import IRModule, relax

from KAS import KernelLoader


def import_templated_model(working_dir: os.PathLike, model_name: str) -> Tuple[IRModule, Tuple]:
    """Import an exported model. Returns the imported model and the input shape."""
    # import the Relax model
    py_mod_name = f"model_relax_{model_name.replace('/', '_')}"
    assert py_mod_name not in sys.modules
    spec = importlib.util.spec_from_file_location(py_mod_name, os.path.join(working_dir, f"{model_name}.py"))
    py_mod = importlib.util.module_from_spec(spec)
    sys.modules[py_mod_name] = py_mod
    spec.loader.exec_module(py_mod)
    relax_mod = py_mod.Module
    assert isinstance(relax_mod, IRModule)
    input_shape = py_mod.INPUT_SHAPE
    return relax_mod, input_shape

def substitute_kernels(relax_mod: IRModule, kernel_loader: KernelLoader) -> IRModule:
    """Substitute the kernels in the imported model."""
    @relax.expr_functor.mutator
    class KernelReplacer(relax.PyExprMutator):
        def __init__(self, mod: IRModule) -> None:
            super().__init__()
            self.mod_ = mod

        def visit_call_(self, call):
            call = self.visit_expr_post_order(call)

            # This is a hack. Since we cannot pass information to Relax, we rewrite the layer to be replaced to this form:
            #   layer(x) -> exp(layer(x - kernel_id))
            # In this way we can know which kernels we should replace and the kernel ids.

            exp_op = tvm.ir.Op.get("relax.exp")
            add_op = tvm.ir.Op.get("relax.add")
            conv2d_op = tvm.ir.Op.get("relax.nn.conv2d")
            subtract_op = tvm.ir.Op.get("relax.subtract")

            # First match the outer exp.
            if call.op != exp_op:
                return call

            original_output = self.lookup_binding(call.args[0])

            logging.info(f"Processing {original_output}...")

            # Relax conv2d does not support bias, and bias is implemented by another add op.
            if original_output.op == add_op:
                has_bias = True
                conv_call = self.lookup_binding(original_output.args[0])
                conv_bias = original_output.args[1]
            elif original_output.op == conv2d_op:
                has_bias = False
                conv_call = original_output
                conv_bias = None
            else:
                assert False, f"Unknown op {original_output.op} when handling a marked kernel!"

            # x - kernel_id
            marked_input = self.lookup_binding(conv_call.args[0])
            assert isinstance(marked_input, relax.Call) and marked_input.op == subtract_op, f"Marked input is illegal: {marked_input}"

            # x
            data = marked_input.args[0]
            logging.info(f"data = {data}")
            conv_weight = conv_call.args[1]
            logging.info(f"conv_weight = {conv_weight}")
            if has_bias:
                logging.info(f"conv_bias = {conv_bias}")

            kernel_id = marked_input.args[1]
            assert isinstance(kernel_id, relax.Constant)
            kernel_id = int(kernel_id.data.numpy())
            logging.info(f"kernel_id = {kernel_id}")

            sinfo_input = data.struct_info
            logging.info(f"sinfo_input = {sinfo_input.shape.values}")
            sinfo_output = original_output.struct_info
            logging.info(f"sinfo_output = {sinfo_output.shape.values}")

            # Restore the kernel. TODO: replace.
            new_output = self.builder_.emit(
                relax.op.nn.conv2d(
                    data,
                    conv_weight,
                    strides=conv_call.attrs.strides,
                    padding=conv_call.attrs.padding,
                    dilation=conv_call.attrs.dilation,
                    groups=conv_call.attrs.groups,
                    data_layout=conv_call.attrs.data_layout,
                    kernel_layout=conv_call.attrs.kernel_layout,
                    out_dtype=conv_call.attrs.out_dtype,
                )
            )
            if has_bias:
                new_output = self.builder_.emit(relax.op.add(new_output, conv_bias))
            logging.info(f"Rewritten.")

            return new_output

            # Proof of concept.
            interesting_bias = relax.const(np.zeros(_shape_expr_to_tuple(sinfo_output.shape), "float32"))
            def te_func(X: te.Tensor, Y: te.Tensor):
                return te.compute(sinfo_output.shape.values, lambda *i: X[i] + Y[i])
            biased = self.builder_.emit_te(te_func, new_output, interesting_bias)

            return biased

        def transform(self) -> IRModule:
            for global_var, func in self.mod_.functions.items():
                if not isinstance(func, relax.Function):
                    continue
                updated_func = self.visit_expr(func)
                self.builder_.update_func(global_var, updated_func)

            return self.builder_.get()

    @tvm.ir.transform.module_pass(opt_level=0, name="KASReplaceKernels")
    class ReplaceKernelsPass:
        """The wrapper for the LowerTensorIR pass."""
        def transform_module(self, mod, ctx):
            return KernelReplacer(mod).transform()

    relax_mod = ReplaceKernelsPass()(relax_mod)
    relax_mod = relax.transform.DeadCodeElimination(["main"])(relax_mod) # To remove markers.
    relax_mod = relax.transform.CanonicalizeBindings()(relax_mod)
    return relax_mod
