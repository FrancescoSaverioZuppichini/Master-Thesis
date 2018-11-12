FROM zuppif/ros-melodic-python3:test

# change this based on your location
ENV TZ=America/Los_Angeles
ENV DEBIAN_FRONTEND noninteractive

RUN ln -s /usr/local/lib/python3.6/dist-packages /usr/local/lib/python3.6/site-packages

RUN mkdir -p root/catkin_ws/src \
&& cd root/catkin_ws/ \ 
&& . /opt/ros/melodic/setup.sh \ 
&& catkin_make

COPY ./resources/catkin_ws_src/webots_ros/ root/catkin_ws/src/webots_ros

RUN cd root/catkin_ws/ \
&& . /opt/ros/melodic/setup.sh \
&& . ./devel/setup.sh \
&& export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages": \
&& catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

RUN /bin/bash -c "echo '. /root/catkin_ws/devel/setup.bash' >> ~/.bashrc"

ADD . /root/code
WORKDIR /root/code

RUN pip3 install -r requirements.txt

# webots
COPY ./resources/src /root/src
RUN apt-get -y update && apt-get install -y  gedit gdebi
RUN apt-get -y update && yes | gdebi /root/src/webots_2018b_amd64.deb

LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}


RUN chmod 777 /root/code/start.sh
SHELL [" ", "-c"]
CMD ["./start.sh"]

#CMD ["python", "main.py"]
