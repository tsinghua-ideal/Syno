from dataclasses import dataclass
from importlib.util import spec_from_file_location, module_from_spec
import logging
import os
import sys
from typing import Callable, Dict, FrozenSet, List, Tuple, Optional

import tvm
from tvm import IRModule, relax

from common import get_specialized_model_name


@dataclass
class KernelMetadata:
    id: int
    sinfo_input: relax.StructInfo
    sinfo_output: relax.StructInfo

def import_templated_model(working_dir: str, model_name: str, batch_size: int, vanilla: bool = False) -> Tuple[IRModule, List[Dict[str, int]], Tuple]:
    """Import an exported model. Returns the imported model, all mappings and the input shape."""
    specialized_mode_name = get_specialized_model_name(model_name, batch_size, vanilla=vanilla)
    # import the Relax model
    py_mod_name = f"model_relax_{specialized_mode_name.replace('/', '_').replace('=', '_')}"
    if py_mod_name not in sys.modules:
        py_mod_path = os.path.join(working_dir, f"{specialized_mode_name}.py")
        spec = spec_from_file_location(py_mod_name, py_mod_path)
        assert spec is not None, f"Failed to load module from {py_mod_path}"
        py_mod = module_from_spec(spec)
        sys.modules[py_mod_name] = py_mod
        spec.loader.exec_module(py_mod)
    else:
        py_mod = sys.modules[py_mod_name]
    relax_mod = py_mod.Module
    assert isinstance(relax_mod, IRModule)
    all_mappings = py_mod.ALL_MAPPINGS
    input_shape = py_mod.INPUT_SHAPE
    return relax_mod, all_mappings, input_shape

def _shape_expr_to_tuple(shape_expr: relax.ShapeExpr) -> tuple:
    return tuple(v.value for v in shape_expr.values)

def substitute_kernels(relax_mod: IRModule, kernel_builder: Optional[Callable[[KernelMetadata, relax.BlockBuilder, relax.Var], relax.Var]]) -> IRModule:
    """
        Substitute the kernels in the imported model.
        If kernel_builder is None, the kernels will be restored as defined in the network.
        Example:
        ```python
            def kernel_builder(kernel_meta: KernelMetadata, bb: relax.BlockBuilder, data: relax.Var) -> relax.Var:
                # This is a simple kernel that adds a constant to the input,
                added = bb.emit(relax.op.add(data, relax.const(kernel_meta.id, "float32")))
                # and performs a 1x1 convolution.
                in_channels = kernel_meta.sinfo_input.shape[1].value
                out_channels = kernel_meta.sinfo_output.shape[1].value
                conv = bb.emit(relax.op.nn.conv2d(
                    added,
                    relax.const(np.zeros((out_channels, in_channels, 1, 1), "float32")),
                    out_dtype="float32",
                ))
                return conv
            result = substitute_kernels(relax_mod, kernel_builder)
            result.show()
        ```
    """
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
            matmul_op = tvm.ir.Op.get("relax.matmul")
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
                bias = original_output.args[1]
                matmul_call = None
            elif original_output.op == conv2d_op:
                has_bias = False
                conv_call = original_output
                bias = None
                matmul_call = None
            elif original_output.op == matmul_op:
                has_bias = False
                conv_call = None
                bias = None
                matmul_call = original_output
            else:
                assert False, f"Unknown op {original_output.op} when handling a marked kernel!"

            # x - kernel_id
            marked_input = self.lookup_binding((conv_call or matmul_call).args[0])
            assert isinstance(marked_input, relax.Call) and marked_input.op == subtract_op, f"Marked input is illegal: {marked_input}"

            # x
            data = marked_input.args[0]
            logging.info(f"data = {data}")

            kernel_id = marked_input.args[1]
            assert isinstance(kernel_id, relax.Constant)
            kernel_id = int(kernel_id.data.numpy())
            logging.info(f"kernel_id = {kernel_id}")

            sinfo_input = data.struct_info
            logging.info(f"sinfo_input = {sinfo_input.shape.values}")
            sinfo_output = original_output.struct_info
            logging.info(f"sinfo_output = {sinfo_output.shape.values}")

            if kernel_builder is None:
                # Restore the kernel.
                if conv_call is not None:
                    conv_weight = conv_call.args[1]
                    logging.info(f"conv_weight = {conv_weight}")
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
                elif matmul_call is not None:
                    matmul_weight = matmul_call.args[1]
                    logging.info(f"matmul_weight = {matmul_weight}")
                    new_output = self.builder_.emit(
                        relax.op.matmul(
                            data,
                            matmul_weight,
                            out_dtype=matmul_call.attrs.out_dtype,
                        )
                    )
                if has_bias:
                    logging.info(f"bias = {bias}")
                    new_output = self.builder_.emit(relax.op.add(new_output, bias))
                logging.info(f"Rewritten.")

                return new_output
            else:
                # Substitute the kernel as given.
                kernel_meta = KernelMetadata(kernel_id, sinfo_input, sinfo_output)
                result = kernel_builder(kernel_meta, self.builder_, data)
                logging.info(f"Rewritten as given in the builder.")
                return result

        def transform(self) -> IRModule:
            for global_var, func in self.mod_.functions.items():
                if not isinstance(func, relax.Function):
                    continue
                updated_func = self.visit_expr(func)
                self.builder_.update_func(global_var, updated_func)
            # TODO: maybe we need to perform some CSE?
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

def construct_kernel_builder(all_mappings: List[Dict[str, int]], generic_weights_builder: Callable[[], List[relax.Constant]], generic_func_builder: Callable[[relax.BlockBuilder, relax.Var], relax.Var]) -> Callable[[KernelMetadata, relax.BlockBuilder, relax.Var], relax.Var]:
    def with_mappings(kernel_meta: KernelMetadata, bb: relax.BlockBuilder, data: relax.Var) -> relax.Var:
        mappings = all_mappings[kernel_meta.id]
        weights = generic_weights_builder(**mappings)
        return generic_func_builder(bb, data, *weights, **mappings)
    return with_mappings

# It seems Relax does not work with Function. It finds it hard to perform fusing passes.
# But it seems Relax will coalesce identical te functions in the first place, so no need to do this.
def _construct_kernel_builder(all_mappings: List[Dict[str, int]], generic_weights_builder: Callable[[], List[relax.Constant]], generic_func_builder: Callable[[relax.BlockBuilder, relax.Var], relax.Var]) -> Callable[[KernelMetadata, relax.BlockBuilder, relax.Var], relax.Var]:
    # There are redundant mappings. We should generate functions for distict mappings, and call the functions.
    # Use frozenset for hash.
    built_functions: Dict[FrozenSet[Tuple[str, int]], relax.GlobalVar] = {}
    def mappings_name(func_key: FrozenSet[Tuple[str, int]]) -> str:
        def tuple_to_str(t: Tuple[str, int]) -> str:
            variable, value = t
            return f"{variable}_{value}"
        return '_'.join(map(tuple_to_str, func_key))
    def with_mappings(kernel_meta: KernelMetadata, bb: relax.BlockBuilder, data: relax.Var) -> relax.Var:
        mappings = all_mappings[kernel_meta.id]
        func_key = frozenset(mappings.items())
        if func_key not in built_functions:
            with bb.function(f"kas_kernel_{mappings_name(func_key)}", private=True):
                inp = relax.Var("inp", kernel_meta.sinfo_input)
                weights = generic_weights_builder(**mappings)
                with bb.dataflow():
                    out = generic_func_builder(bb, inp, *weights, **mappings)
                    df_out = bb.emit_output(out)
                built_function = bb.emit_func_output(df_out, params=[inp])
                logging.info(f"Built function {built_function}")
            built_functions[func_key] = built_function
        else:
            built_function = built_functions[func_key]
        return bb.emit(relax.Call(built_function, [data]))
    return with_mappings
