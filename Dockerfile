FROM docker pull pytorch/pytorch:latest


RUN apt update && apt install -y \
    git \
    fish \
    htop \
    man \
    shellcheck \
    tar \
    tmux \
    wget \
    vim \
    && apt clean

COPY requirements.txt /install/requirements.txt
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir -r /install/requirements.txt

COPY vimrc /install/.vimrc

RUN mkdir -p ~/.vim/undodir && \
    mkdir -p ~/.vim/pack/plug/start && \
    cd ~/.vim/pack/plug/start && \
    git clone https://github.com/morhetz/gruvbox.git && \
    git clone https://github.com/scrooloose/nerdtree.git

RUN wget --quiet https://dl.min.io/client/mc/release/linux-amd64/mc -O /bin/mc && \
    chmod +x /bin/mc
