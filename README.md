# TensorRT discrete Fourier transform (DFT) plugins

This project contains several TensorRT plugins which implement certain operators, such as discrete Fourier transform (DFT) along with its inverse. These operators are used in deep learning models that utilize, for example, Fourier Neural Operators (FNO), such as [FourCastNet](https://docs.nvidia.com/deeplearning/modulus/user_guide/neural_operators/fourcastnet.html). TensorRT currently does not provide support for such operators. Additionally, support for [ONNX DFT](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DFT) operator is limited or not available in some of the frameworks, such as PyTorch. This library enables export of models that use DFT to TensorRT for fast inference.

The plugins are created using [TensorRT C++ Plugin API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#add_custom_layer) and can be used to export ONNX models to TensorRT and perform inference with the help of [C++](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics) or [Python](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics) client APIs.

In addition to TensorRT plugins, the package provides a convenience Python wrapper function to load all currently implemented plugins into memory for use by the inference code.

## Supported operators

* RFFT and RFFT2, which correspond to [torch.fft.rfft](https://pytorch.org/docs/1.13/generated/torch.fft.rfft.html) and [torch.fft.rfft2](https://pytorch.org/docs/1.13/generated/torch.fft.rfft2.html) respectively.
    PyTorch operators must be exported to ONNX using [ONNX Contrib Rfft](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Rfft) operator.
* IRFFT and IRFFT2, which correspond to [torch.fft.irfft](https://pytorch.org/docs/1.13/generated/torch.fft.irfft.html) and [torch.fft.irfft2](https://pytorch.org/docs/1.13/generated/torch.fft.irfft2.html) respectively.
    PyTorch operators must be exported to ONNX using [ONNX Contrib Irfft](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Irfft) operator.

## Building the code

### Docker image

It is recommended to build a Docker image using provided [Dockerfile](./docker/Dockerfile) and build script:
```
cd ./docker
./build_image.sh
```
The image provides all components necessary to build and run the code.

However, if building an image is not feasible, then a user needs to make sure that all necessary components are installed. In particular, the code depends on PyTorch, TensorRT and CMake.

### Building the library

#### Build from Docker
Once the Docker image is built, the [build script](./build_with_docker.sh) can be used to build the library in Docker container as well as run the tests:
```
./build_with_docker.sh
```

The script creates a Docker container and runs `pip install` followed by tests.

#### Build without Docker
If Docker is not used, then the library can be built and installed using the following commands:
```
pip install --user -e .
```
This command installs the package in [editable mode](https://pip.pypa.io/en/latest/topics/local-project-installs/#editable-installs).


To verify the installation, it is recommended to run the package unit tests:
```
pytest .
```

To uninstall the package, simply run:
```
pip uninstall trt-dft-plugins
```

## Using the plugins

When a model uses one or more operators that require custom plugins, there are usually 2 steps needed to enable the model inference with TensorRT.

1. Export the model from ONNX format to TensorRT plan. This step is usually performed by [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) utility and requires specifying a path to plugin `.so` shared library:

    ```
    trtexec --buildOnly --onnx=path_to_ONNX_file --saveEngine=path_to_resulting_TRT_plan --plugins=./src/trt_dft_plugins/libtrt_dft_plugins.so
    ```
    (make sure the path to `libtrt_dft_plugins.so` is correct)


    This command will build and save TensorRT plan from provided ONNX file. There might be other options required, such as `--shapes`, but those are model-dependent.

    Once the TensorRT plan is built, it can be used to run inference and measure performance using the same `trtexec` tool:

    ```
    trtexec --loadEngine=path_to_TRT_plan --plugins=./src/trt_dft_plugins/libtrt_dft_plugins.so
    ```

2. Load and use the model from C++ or Python client code.

    In this case, plugin's `.so` shared library needs to be loaded before running inference. In case of Python, a convenience function, `trt_dft_plugins.load_plugins` can be used to load all currently supported plugins.

    See the [unit tests code](./tests/test_dft.py) for more details.
