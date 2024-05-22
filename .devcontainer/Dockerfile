FROM nvcr.io/nvidia/tensorrt:23.05-py3

# opencv
RUN apt-get update
RUN apt-get install libopencv-dev -y
RUN apt-get install libglib2.0 libglib2.0-dev libssl-dev -y

# redis client-server
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

# hiredis and json 
RUN mkdir /opt/tensorrt/hiredis 
RUN mkdir /opt/tensorrt/nlohmann 

RUN git clone https://github.com/redis/hiredis.git /opt/tensorrt/hiredis 
RUN git clone https://github.com/nlohmann/json.git /opt/tensorrt/nlohmann 

RUN mkdir /opt/tensorrt/hiredis/build && cd /opt/tensorrt/hiredis/build && cmake /opt/tensorrt/hiredis && make -j$(nproc)
RUN apt-get install libhiredis-dev -y

RUN mkdir /opt/tensorrt/nlohmann/build && cd /opt/tensorrt/nlohmann/build && cmake /opt/tensorrt/nlohmann && make -j$(nproc)
RUN mv /opt/tensorrt/nlohmann/include/nlohmann /usr/include 
RUN apt-get install libevent-dev -y
