FROM zuppif/ros-melodic-python3:test

RUN ln -s /usr/local/lib/python3.6/dist-packages /usr/local/lib/python3.6/site-packages

RUN mkdir -p root/catkin_ws/src \
&& cd root/catkin_ws/ \ 
&& . /opt/ros/melodic/setup.sh \ 
&& catkin_make

RUN git clone https://github.com/romarcg/krock-sim.git \
&& cp -r krock-sim/catkin_ws_src/webots_ros/ root/catkin_ws/src/webots_ros

RUN cd root/catkin_ws/ \
&& . /opt/ros/melodic/setup.sh \
&& . ./devel/setup.sh \
&& export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages": \
&& catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 \
&& pip3 install defusedxml

RUN /bin/bash -c "echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc"
RUN /bin/bash -c "echo 'source /root/catkin_ws/devel/setup.bash' >> ~/.bashrc"

ADD . /root/code
WORKDIR /root/code

RUN pip3 install -r requirements.txt

RUN chmod 777 /root/code/start.sh
SHELL [" ", "-c"]
#CMD ['source /opt/ros/melodic/setup.bash && source /root/catkin_ws/devel/setup.bash']
#CMD ["./start.sh"]
