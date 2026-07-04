FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    MUJOCO_GL=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    cmake \
    ca-certificates \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglfw3-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

RUN conda create -n robomimic_venv python=3.9 -y

RUN conda run -n robomimic_venv pip install --upgrade pip setuptools wheel

RUN conda run -n robomimic_venv pip install \
    torch==2.0.0+cu118 \
    torchvision==0.15.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN conda run -n robomimic_venv pip install \
    numpy \
    scipy \
    h5py \
    tqdm \
    tensorboard \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    matplotlib \
    pandas \
    termcolor \
    psutil \
    pyyaml \
    easydict \
    wandb \
    gym==0.26.2

WORKDIR /workspace

COPY . /workspace

RUN conda run -n robomimic_venv pip install -e ./robomimic && \
    cd /workspace/third_party/robosuite && \
    conda run -n robomimic_venv pip install -r requirements.txt && \
    conda run -n robomimic_venv pip install -e .

CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && bash"]