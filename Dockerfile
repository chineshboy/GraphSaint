FROM    nvidia/cuda:11.0-runtime-ubuntu20.04
MAINTAINER      chineshboy "chineshboy@hotmail.com"

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

# Locale ENV
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 HOME_DIR=/data

COPY requirements.txt /tmp/requirements.txt
# create timezone info to skip interactive data requests
ENV TZ=Asia/Hong_Kong
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    addgroup --gid ${GROUP_ID} ${USER_NAME} &&\
    useradd -l -u ${USER_ID} -g ${USER_NAME} -p '*' -s /bin/bash ${USER_NAME} &&\
    install -d -m 0755 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME} \
    ;fi \
    && apt update \
    && apt install -y ca-certificates \
    && update-ca-certificates \
    && apt install -y apt-utils locales && locale-gen en_US.UTF-8 \
    && apt install -y python3-pip curl wget vim \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html \
    && cat /tmp/requirements.txt | xargs -n 1 -L 1 pip3 install

CMD [ "/bin/bash" ]
