FROM ros-melodic-python3

#RUN adduser --disabled-password --gecos '' newuser
#USER newuser
# RUN  echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc &&  /bin/bash -c "source ~/.bashrc"
RUN ln -s /usr/local/lib/python3.6/dist-packages /usr/local/lib/python3.6/site-packages

# You need to always source before calling catkin_make
RUN export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages"
RUN mkdir -p root/catkin_ws/src \
&& cd root/catkin_ws/ \ 
&& . /opt/ros/melodic/setup.sh \ 
&& catkin_make

COPY ./webots_ros/ root/catkin_ws/src/webots_ros

RUN cd root/catkin_ws/ \
&& . /opt/ros/melodic/setup.sh \
&& . ./devel/setup.sh \
&& export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages" \
&& catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

RUN /bin/bash -c "echo '. /root/catkin_ws/devel/setup.bash' >> ~/.bashrc"

ADD . /root/code
WORKDIR /root/code

RUN pip3 install -r requirements.txt
RUN chmod 777 /root/code/start.sh
SHELL ["/bin/bash", "-c"]
# RUN ./start.sh
# RUN python3 main.py
CMD ["./start.sh"]