# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import io
from typing import Optional
import pytest

import tensorrt as trt
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn

from trt_dft_plugins import load_plugins


# This is a simple, but limited, implementation of RFFT/IRFFT custom ops
# which does not require external dependencies. Its purpose is to test TRT export pipeline.
class OnnxRfft2(Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> torch.Value:
        return torch.view_as_real(torch.fft.rfft2(input, dim=(-2, -1), norm="backward"))

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        return g.op(
            "com.microsoft::Rfft", input, normalized_i=0, onesided_i=1, signal_ndim_i=2
        )


class OnnxIrfft2(Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> torch.Value:
        return torch.fft.irfft2(
            torch.view_as_complex(input), dim=(-2, -1), norm="backward"
        )

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        return g.op(
            "com.microsoft::Irfft", input, normalized_i=0, onesided_i=1, signal_ndim_i=2
        )


@pytest.fixture(scope="session", autouse=True)
def load_trt_plugins():
    load_plugins()


@pytest.fixture()
def trt_logger():
    return trt.Logger(trt.Logger.WARNING)


def export_to_onnx(
    model: nn.Module, inp: Tensor, verbose: Optional[bool] = True
) -> bytes:
    with io.BytesIO() as onnx_model:
        # Export to ONNX.
        torch.onnx.export(
            model,
            inp,
            onnx_model,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            opset_version=15,
            verbose=verbose,
        )
        return onnx_model.getvalue()


def build_trt_plan(onnx_model: bytes, logger: trt.ILogger) -> trt.IHostMemory:
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse(onnx_model)
    assert success, "\n".join(
        str(parser.get_error(i)) for i in range(parser.num_errors)
    )

    config = builder.create_builder_config()
    return builder.build_serialized_network(network, config)


def run_trt_inference(
    trt_plan: trt.IHostMemory, x: Tensor, y: Tensor, logger: trt.ILogger
) -> Tensor:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(trt_plan)
    context = engine.create_execution_context()
    x = x.cuda()
    y_device = y.device
    y = y.cuda()
    buffers = [x.data_ptr(), y.data_ptr()]
    context.execute_v2(buffers)
    return y.to(y_device)


def test_plugins_load():
    loaded_plugins = {p.name for p in trt.get_plugin_registry().plugin_creator_list}
    assert "Rfft" in loaded_plugins
    assert "Irfft" in loaded_plugins


@pytest.mark.parametrize("dft_dim1", [1, 2])
@pytest.mark.parametrize("dft_dim2", [4])
@pytest.mark.parametrize("num_c", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_rfft2(trt_logger, dft_dim1, dft_dim2, num_c, batch_size):
    class RfftModel(nn.Module):
        def forward(self, x):
            return OnnxRfft2.apply(x)

    model = RfftModel()

    torch.manual_seed(1)
    x = torch.randn(batch_size, num_c, dft_dim1, dft_dim2)

    # 1. Export to ONNX.
    onnx_model = export_to_onnx(model, x)

    # 2. Build TRT plan from ONNX.
    trt_plan = build_trt_plan(onnx_model, trt_logger)

    # 3. Run TRT inference.
    y_expected = OnnxRfft2.apply(x)
    # y stores the output of the RFFT2 ops.
    y = torch.empty_like(y_expected)
    y = run_trt_inference(trt_plan, x, y, trt_logger)

    # Both implementations should produce the same result.
    assert torch.allclose(y_expected, y)


@pytest.mark.parametrize("dft_dim1", [1, 2])
@pytest.mark.parametrize("dft_dim2", [4])
@pytest.mark.parametrize("num_c", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_irfft2(trt_logger, dft_dim1, dft_dim2, num_c, batch_size):
    class IrfftModel(nn.Module):
        def forward(self, x):
            return OnnxIrfft2.apply(x)

    model = IrfftModel()

    torch.manual_seed(1)
    x = torch.randn(batch_size, num_c, dft_dim1, dft_dim2)

    # Compute RFFT first.
    y = OnnxRfft2.apply(x)

    # 1. Export to ONNX.
    onnx_model = export_to_onnx(model, y)

    # 2. Build TRT plan from ONNX.
    trt_plan = build_trt_plan(onnx_model, trt_logger)

    # 3. Run TRT inference.
    x_expected = OnnxIrfft2.apply(y)
    # x_actual stores the output of the IRFFT2 ops.
    x_actual = torch.empty_like(x_expected)
    x_actual = run_trt_inference(trt_plan, y, x_actual, trt_logger)

    # Both implementations should produce the same result.
    assert torch.allclose(x_expected, x_actual)
