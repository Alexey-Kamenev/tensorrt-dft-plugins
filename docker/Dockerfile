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

FROM nvcr.io/nvidia/pytorch:22.11-py3

ARG USERNAME=du
ARG USER_UID
ARG USER_GID

ENV DEBIAN_FRONTEND=noninteractive

# Add a non-root user with a fixed UID and GID
RUN set -eux; \
    groupadd --gid $USER_GID $USERNAME; \
    useradd --uid $USER_UID --gid $USER_GID --no-log-init -m -G video $USERNAME;

RUN set -ex; \
    apt-get update;

RUN pip install --upgrade --no-cache-dir \
        black       \
        pip         \
        setuptools

# Need to uninstall cmake package as it does not work properly.
RUN pip uninstall -y cmake

WORKDIR /workspace

# Download and install CMake.
RUN set -ex; \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3-linux-x86_64.sh; \
    chmod +x ./cmake-3.25.3-linux-x86_64.sh;    \
    mkdir /opt/cmake; \
    ./cmake-3.25.3-linux-x86_64.sh --skip-license --prefix=/opt/cmake; \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# Add sudo and allow the non-root user to execute commands
# as root without a password.
RUN set -ex; \
    apt-get install -y --no-install-recommends \
        sudo;   \
    rm -rf /var/lib/apt/lists/*; \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME;
