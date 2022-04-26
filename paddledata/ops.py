import os
import paddle
from paddle.fluid.layers import utils
from paddle.fluid.layer_helper import LayerHelper

cwd = os.path.dirname(os.path.abspath(__file__))
decoding_lib = os.path.join(cwd, "libimage_decode_op.so")

ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)


def decode_random_crop(x,
                    num_threads=2,
                    host_memory_padding=0,
                    device_memory_padding=0,
                    aspect_ratio_min=3./4.,
                    aspect_ratio_max=4./3.,
                    area_min=0.08,
                    area_max=1.,
                    num_attempts=10):
    # prepare inputs and outputs
    local_rank = paddle.distributed.get_rank()
    program_id = utils._hash_with_id(mode, num_threads, "custom_decode", local_rank)
    ins = {'X@VECTOR' : x}
    attrs = {
             "num_threads": num_threads,
             "local_rank": local_rank,
             "program_id": program_id,
             "host_memory_padding": host_memory_padding,
             "device_memory_padding": device_memory_padding,
             "aspect_ratio_min": aspect_ratio_min,
             "aspect_ratio_max": aspect_ratio_max,
             "area_min": area_min,
             "area_max": area_max}
    outs = {}
    out_names = ['Out@VECTOR']

    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if paddle.fluid.framework.in_dygraph_mode():
        if paddle.fluid.framework._in_eager_mode():
        # if _in_eager_mode():
            ctx = core.CustomOpKernelContext()
            for i in [x]:
                ctx.add_inputs(i)
            for j in []:
                ctx.add_attr(j)
            for out_name in out_names:
                outs[out_name] = [core.eager.Tensor() for _ in range(len(x))]
                ctx.add_outputs(outs[out_name])
            core.eager._run_custom_op(ctx, "custom_decode", True)
        else:
            for out_name in out_names:
                outs[out_name] = [VarBase() for _ in range(len(x))]
            _dygraph_tracer().trace_op(type="custom_decode", inputs=ins, outputs=outs, attrs=attrs)
    else:
        helper = LayerHelper("custom_decode", **locals())
        x_dtype = x[0].dtype
        for out_name in out_names:
            outs[out_name] = [helper.create_variable(dtype=x_dtype) for _ in range(len(x))]

        helper.append_op(type="custom_decode", inputs=ins, outputs=outs, attrs=attrs)

    res = [outs[out_name] for out_name in out_names]

    return res[0] if len(res)==1 else res
