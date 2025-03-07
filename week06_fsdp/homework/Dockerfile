FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
RUN useradd -U -m "$USERNAME" \
    && echo "$USERNAME" ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/"$USERNAME" \
    && chmod 0440 /etc/sudoers.d/"$USERNAME"
