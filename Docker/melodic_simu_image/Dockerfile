
FROM nvcr.io/nvidia/cudagl:10.1-devel-ubuntu18.04

# Setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Setup environment
RUN apt-get update
RUN apt-get upgrade -y --no-install-recommends
RUN DEBIAN_FRONTEND=noninteractive

# Install preliminary packages
RUN apt-get update
RUN apt-get upgrade -y --no-install-recommends
RUN apt-get install gcc g++ python python-pip python-dev vtk7 -y --no-install-recommends
RUN apt-get install gedit firefox wget git xvfb vim nano -y --no-install-recommends
RUN apt-get install curl gnupg2 lsb-release -y --no-install-recommends
RUN apt-get update
RUN apt-get upgrade -y --no-install-recommends

# Install other packages
RUN apt-get install ffmpeg -y --no-install-recommends
RUN apt-get install xdg-utils -y --no-install-recommends
RUN apt-get install vim-gtk -y --no-install-recommends

# Install preliminary python packages
RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install numpy scikit-learn matplotlib PyYAML
RUN pip install psutil 
RUN pip install tabulate
RUN pip install moviepy
RUN pip install edt
RUN pip install plyfile scipy jupyterlab

# Install ros melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN apt-get install ros-melodic-desktop-full -y --no-install-recommends

# Install packages
RUN apt-get install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential -y --no-install-recommends
RUN rosdep init
RUN rosdep update
RUN apt-get update
RUN apt-get install apt-utils -y --no-install-recommends 
RUN apt-get install python-catkin-tools -y --no-install-recommends
RUN apt-get install ros-melodic-velodyne* -y --no-install-recommends
RUN apt-get install ros-melodic-jackal* -y --no-install-recommends
RUN apt-get install ros-melodic-pointcloud-to-laserscan -y --no-install-recommends
RUN apt-get install ros-melodic-rgbd-launch -y --no-install-recommends
RUN apt-get install ros-melodic-range-sensor-layer -y --no-install-recommends
RUN apt-get install ros-melodic-openni2* -y --no-install-recommends
RUN apt-get install ros-melodic-octomap* -y --no-install-recommends
RUN apt-get install ros-melodic-ros-control -y --no-install-recommends
RUN apt-get install ros-melodic-tf2 -y --no-install-recommends
RUN apt-get install ros-melodic-tf2-py -y --no-install-recommends
RUN apt-get install ros-melodic-tf2-kdl -y --no-install-recommends
RUN apt-get install ros-melodic-tf2-ros -y --no-install-recommends
RUN apt-get install ros-melodic-tf2-msgs -y --no-install-recommends
RUN apt-get install ros-melodic-tf2-eigen -y --no-install-recommends
RUN apt-get install ros-melodic-tf2-sensor-msgs -y --no-install-recommends
RUN apt-get install ros-melodic-tf2-geometry-msgs -y --no-install-recommends
RUN apt-get install ros-melodic-people-msgs -y --no-install-recommends
RUN apt-get install ros-melodic-people-velocity-tracker -y --no-install-recommends
RUN apt-get install ros-melodic-navigation* -y --no-install-recommends
RUN apt-get install ros-melodic-people-msgs -y --no-install-recommends
RUN apt-get install ros-melodic-social-navigation-layers -y --no-install-recommends
RUN apt-get install ros-melodic-teleop-twist-keyboard -y --no-install-recommends
RUN apt-get install ros-melodic-costmap-converter -y --no-install-recommends
RUN apt-get install ros-melodic-costmap-2d -y --no-install-recommends
# RUN apt-get install ros-melodic-cmake-modules -y --no-install-recommends
# RUN apt-get install ros-melodic-dynamic-reconfigure -y --no-install-recommends
# RUN apt-get install ros-melodic-geometry-msgs -y --no-install-recommends
# RUN apt-get install ros-melodic-interactive-markers -y --no-install-recommends
# RUN apt-get install ros-melodic-message-generation -y --no-install-recommends
# RUN apt-get install ros-melodic-nav-core -y --no-install-recommends
# RUN apt-get install ros-melodic-nav-msgs -y --no-install-recommends
RUN apt-get install ros-melodic-mbf-costmap-core -y --no-install-recommends
RUN apt-get install ros-melodic-mbf-msgs -y --no-install-recommends
RUN apt-get install ros-melodic-libg2o -y --no-install-recommends
#RUN apt-get update && apt-get install -y libgazebo9-dev
RUN apt-get install -y gnome-core --no-install-recommends
RUN apt-get install -y mesa-utils --no-install-recommends

# Install Gazebo
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys D2486D2DD83DB69272AFE98867170598AF249743
RUN apt-get update
RUN apt-get upgrade -y --no-install-recommends

# Install other stuff
RUN apt-get update
RUN apt-get install psmisc -y --no-install-recommends
RUN apt-get install ros-melodic-ros-numpy -y --no-install-recommends
RUN apt-get install python-tk ipython -y --no-install-recommends
RUN apt-get install xterm -y --no-install-recommends
ENV TERM xterm-256color

# Install jsk for melodic
RUN apt-get update
RUN apt-get install ros-melodic-jsk-rviz-plugins -y --no-install-recommends

# Setup user information
ARG username0=bag
ARG userid0=1085
RUN useradd -ms /bin/bash -u $userid0 $username0
RUN adduser $username0 root
RUN adduser $username0 sudo
RUN echo $username0':myhaljack' | chpasswd
USER $username0

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN echo "export GAZEBO_RESOURCE_PATH=/home/$username0/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/src/myhal_simulator/models" >> ~/.bashrc
RUN echo "export GAZEBO_PLUGIN_PATH=/home/$username0/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/devel/lib" >> ~/.bashrc
RUN echo "export GAZEBO_MODEL_PATH=/home/$username0/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/src/myhal_simulator/models" >> ~/.bashrc
RUN cat /usr/share/gazebo/setup.sh >> ~/.bashrc

ENV GAZEBO_RESOURCE_PATH=/home/$username0/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/src/myhal_simulator/models
ENV GAZEBO_PLUGIN_PATH=/home/$username0/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/devel/lib
ENV GAZEBO_MODEL_PATH=/home/$username0/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/src/myhal_simulator/models

ENV GAZEBO_RESOURCE_PATH=/usr/share/gazebo-9:${GAZEBO_RESOURCE_PATH}
ENV GAZEBO_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-9/plugins:${GAZEBO_PLUGIN_PATH}
ENV GAZEBO_MODEL_PATH=/usr/share/gazebo-9/models:${GAZEBO_MODEL_PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu/gazebo-9/plugins
ENV OGRE_RESOURCE_PATH=/usr/lib/x86_64-linux-gnu/OGRE-1.9.0

# nvidia-docker hooks
#LABEL com.nvidia.volumes.needed="nvidia_driver"
#ENV PATH /usr/local/nvidia/bin:${PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Some QT-Apps/Gazebo don't show controls without this
ENV QT_X11_NO_MITSHM 1

ENV ROS_DISTRO melodic
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV USER=$username0

RUN rosdep update

# Set xterm defaults
WORKDIR /home/$username0
RUN echo "xterm*background: black" >> .Xresources
RUN echo "xterm*foreground: lightgray" >> .Xresources
RUN echo "xterm*faceName: monospace:pixelsize=14" >> .Xresources

CMD ["bash"]
