#!/bin/bash

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

CONTAINER_NAME="trt-dft-plugins"
IMAGE_NAME="trt-dft-plugins"
IMAGE_TAG="23.03"
CONTAINER_DIR="/workspace/${CONTAINER_NAME}"

docker run -it --gpus all                             \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --network=host                                    \
    -v `realpath .`:${CONTAINER_DIR}:rw \
    -u $(id -u):$(id -g)                              \
    --ulimit memlock=-1                               \
    --ulimit stack=67108864                           \
    --name=${CONTAINER_NAME}                          \
    --rm                                              \
    ${IMAGE_NAME}:${IMAGE_TAG}                        \
    bash -c "cd ${CONTAINER_DIR} && pip install --user -e . && pytest ."
