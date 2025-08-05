FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Use an appropriate base image for your algorithm.
# As an example, we use the official pytorch image.

# In reality this baseline algorithm only needs numpy so we could use a smaller image:
#FROM --platform=linux/amd64 python:3.11-slim


# Ensure that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# Install requirements.txt
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

#RUN pip install --upgrade pip
#RUN python -m pip install \
#    --user \
#    --no-cache-dir \
#    --no-color \
#    --requirement /opt/app/requirements.txt

#if command -v wget &> /dev/null; then
#elif command -v curl &> /dev/null; then
#    CMD="curl -L -O"
#else
#    echo "Please install wget or curl to download the checkpoints."
#    exit 1
#fi
#
#SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
##sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
##sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
##sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
#sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
#
## SAM 2.1 checkpoints
#$CMD $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; exit 1; }
#$CMD $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url"; exit 1; }
#$CMD $sam2p1_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2p1_hiera_b_plus_url"; exit 1; }

#RUN apt-get update && apt-get install -y wget
RUN pip install gdown
#RUN wget -P /opt/app/resources/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
#RUN wget -P /opt/app/resources/ https://drive.google.com/file/d/1myBSBdS6wpN-9c0r7poY23_XaagHaEh-/view?usp=drive_link
#RUN wget -P /opt/app/resources/ https://drive.google.com/file/d/1-3EQacNVMiUVyE6Gebaw5f6Kr1t0T7xA/view?usp=drive_link
#RUN wget -P /opt/app/resources/ https://drive.google.com/file/d/1ciXB2eSZXrS-zRu7GtMGuJShNSuQkqNt/view?usp=drive_link
RUN gdown -O /opt/app/resources --id 1myBSBdS6wpN-9c0r7poY23_XaagHaEh-
RUN gdown -O /opt/app/resources --id 1-3EQacNVMiUVyE6Gebaw5f6Kr1t0T7xA
RUN gdown -O /opt/app/resources --id 1ciXB2eSZXrS-zRu7GtMGuJShNSuQkqNt

#COPY --chown=user:user sam2.1_hiera_large.pt /opt/app/resources/sam2.1_hiera_large.pt
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user inference_model.py /opt/app/
COPY --chown=user:users dam4sam /opt/app/dam4sam
COPY --chown=user:user sam2 /opt/app/sam2
COPY --chown=user:user training /opt/app/training
COPY --chown=user:user setup.py /opt/app/

RUN uv pip install -r requirements.txt --system
RUN uv pip uninstall typing-extensions --system
RUN uv pip install typing_extensions --system

# TODO why cant falsh_attn be installed?
# TODO check all versions and remove unneeded ones
# TODO use org sam2

#RUN pip install flash_attn

# Add any other files that are needed for your algorithm
# COPY --chown=user:user <source> <destination>
#RUN pip install SimpleITK
#RUN pip install hydra-Core
#RUN pip install vot-toolkit
#RUN pip install vot-trax
USER user
ENTRYPOINT ["python", "inference.py"]