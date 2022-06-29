#FROM osrf/ros:noetic-desktop-full
#FROM vistart/cuda:10.1-cudnn7-runtime-ubuntu20.04
FROM nvcr.io/nvidia/cudagl:11.0-devel-ubuntu20.04

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
RUN apt-get install gcc g++ python3 python3-pip python3-dev vtk7 -y --no-install-recommends
RUN apt-get install gedit firefox wget git xvfb vim nano -y --no-install-recommends
RUN apt-get update
RUN apt-get upgrade -y --no-install-recommends

# Install preliminary python packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install numpy scikit-learn matplotlib PyYAML

# Install pypcd
RUN pip3 install --upgrade git+https://github.com/klintan/pypcd.git
RUN pip3 install imageio

# Install PyTorch
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Additionnal stuff
RUN apt install ffmpeg -y --no-install-recommends
RUN pip3 install imageio imageio-ffmpeg

# Install ROS 
RUN apt-get update
RUN apt-get install curl gnupg2 lsb-release -y --no-install-recommends
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt update
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update

RUN apt-get update
RUN apt-get install ros-noetic-desktop-full -y --no-install-recommends

# Install ROS packages
RUN apt-get install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y --no-install-recommends
RUN rosdep init
RUN rosdep update

RUN apt-get install python3-catkin-tools -y --no-install-recommends
RUN apt-get install ros-noetic-velodyne* -y --no-install-recommends
RUN apt-get install ros-noetic-pointcloud-to-laserscan -y --no-install-recommends
RUN apt-get install ros-noetic-rgbd-launch -y --no-install-recommends
# RUN apt-get install ros-noetic-range-sensor-layer -y --no-install-recommends
RUN apt-get install ros-noetic-openni2* -y --no-install-recommends
RUN apt-get install ros-noetic-octomap* -y --no-install-recommends
RUN apt-get install ros-noetic-ros-control -y --no-install-recommends
RUN apt-get install ros-noetic-tf2* -y --no-install-recommends
RUN apt-get install ros-noetic-people-msgs -y --no-install-recommends
RUN apt-get install ros-noetic-people-velocity-tracker -y --no-install-recommends
RUN apt-get install ros-noetic-navigation* -y --no-install-recommends
# RUN apt-get install ros-noetic-social-navigation-layers -y --no-install-recommends
RUN apt-get install ros-noetic-teleop-twist-keyboard -y --no-install-recommends

# Additionnal packages
RUN apt-get install -y gnome-core mesa-utils --no-install-recommends

# Install Gazebo
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys D2486D2DD83DB69272AFE98867170598AF249743
RUN apt-get update
RUN apt-get upgrade -y --no-install-recommends

# Additionnal packages
RUN apt-get install psmisc -y --no-install-recommends
RUN apt-get install ros-noetic-ros-numpy -y --no-install-recommends
RUN pip3 install plyfile scipy jupyterlab

# Install gazebo
RUN apt-get install gazebo11
RUN apt-get install libgazebo11-dev -y --no-install-recommends
RUN apt-get install ros-noetic-gazebo* -y --no-install-recommends

# Additionnal packages
RUN apt-get install python3-tk -y --no-install-recommends
RUN pip3 install ipython
RUN pip3 install psutil
RUN pip3 install tabulate
RUN apt-get install ffmpeg -y --no-install-recommends
RUN apt-get install xdg-utils -y --no-install-recommends
RUN apt-get install vim-gtk -y --no-install-recommends
RUN pip3 install moviepy

RUN apt-get install python3-catkin-tools python3-osrf-pycommon -y --no-install-recommends

RUN apt-get install ros-noetic-robot-localization -y --no-install-recommends
RUN apt-get install ros-noetic-twist-mux -y --no-install-recommends
RUN apt-get install ros-noetic-hector-gazebo-plugins -y --no-install-recommends
RUN apt-get install ros-noetic-python-qt-binding -y --no-install-recommends
RUN apt-get install ros-noetic-qt-gui-core -y --no-install-recommends
RUN apt-get install ros-noetic-rqt-graph -y --no-install-recommends
RUN apt-get install ros-noetic-costmap-converter -y --no-install-recommends

RUN pip3 install pynvml

# RUN apt-get install libnvidia-gl-430:i386

# Install Mayavi from source
#RUN pip3 install --upgrade PyQt5
WORKDIR /workspace/compiled_libs
RUN git clone --recursive --progress https://github.com/enthought/mayavi.git
WORKDIR /workspace/compiled_libs/mayavi
RUN pip3 install -r requirements.txt
RUN python3 setup.py install 

# Install mayavi and QT
#RUN apt-get install pyqt5-dev-tools pyqt5-dev -y --no-install-recommends

RUN pip3 install open3d
RUN pip3 install autopep8 flake8

# RUN export QT_PLUGIN_PATH=/usr/lib/qt/plugins
# RUN echo "export QT_PLUGIN_PATH=/usr/lib/qt/plugins" >> ~/.bashrc

# RUN export ETS_TOOLKIT=pyqt5
# RUN echo "export ETS_TOOLKIT=pyqt5" >> ~/.bashrc

# Install dependencies for g2o
RUN apt-get install -y --no-install-recommends \
    libsuitesparse-dev \
    qtdeclarative5-dev \
    qt5-qmake \
    libqglviewer-dev-qt5 \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev 

# Install ceres
WORKDIR /workspace/compiled_libs
RUN wget http://ceres-solver.org/ceres-solver-2.0.0.tar.gz
RUN tar zxf ceres-solver-2.0.0.tar.gz
RUN mkdir ceres-bin
WORKDIR /workspace/compiled_libs/ceres-bin
RUN cmake ../ceres-solver-2.0.0
RUN make -j8
RUN make test
RUN make install


# Other pip install
RUN pip3 install pyvista


# Setup user information
ARG username0=bag
ARG userid0=1085
RUN useradd -ms /bin/bash -u $userid0 $username0
RUN adduser $username0 root
RUN adduser $username0 sudo
RUN echo $username0':password' | chpasswd
USER $username0
WORKDIR /home/$username0/catkin_ws


RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "export USER=$username0" >> ~/.bashrc
RUN echo "export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:/home/$username0/catkin_ws/src/myhal_simulator/models" >> ~/.bashrc
RUN echo "export GAZEBO_RESOURCE_PATH=${GAZEBO_RESOURCE_PATH}:/home/$username0/catkin_ws/src/myhal_simulator/models" >> ~/.bashrc
RUN echo "export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:/home/$username0/catkin_ws/devel/lib" >> ~/.bashrc

# nvidia-docker hooks
#LABEL com.nvidia.volumes.needed="nvidia_driver"
#ENV PATH /usr/local/nvidia/bin:${PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Some QT-Apps/Gazebo don't show controls without this
ENV QT_X11_NO_MITSHM 1

ENV ROS_DISTRO noetic
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV USER=$username0


