FROM nvcr.io/nvidia/tensorrt:23.05-py3

# opencv
RUN apt-get update
RUN apt-get install libopencv-dev -y
RUN apt-get install libglib2.0 libglib2.0-dev libssl-dev -y

# redis client-server
RUN apt-get install libevent-dev
RUN apt install lsb-release curl gpg -y 
RUN apt-get install language-pack-id -y
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
RUN apt-get install redis -y 

ENV LANGUAGE=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
RUN locale-gen en_US.UTF-8
RUN dpkg-reconfigure locales