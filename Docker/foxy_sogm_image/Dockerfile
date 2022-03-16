#
# this dockerfile roughly follows the 'Ubuntu install of ROS Melodic' from:
#   http://wiki.ros.org/melodic/Installation/Ubuntu

#
ARG BASE_IMAGE=nvcr.io/nvidia/cudagl:11.3.0-devel-ubuntu20.04
# ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.10-py3
FROM ${BASE_IMAGE}

ARG ROS_PKG=desktop_full
ENV ROS_DISTRO=foxy
ENV ROS_ROOT=/opt/ros/foxy

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"] 

WORKDIR /tmp

# change the locale from POSIX to UTF-8
RUN apt-get update && apt-get install -y locales
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        python3 \
        python3-pip \
        python3-dev \
        vtk7 \
        gedit \
        firefox \
        wget \
        vim \
        nano \
        git \
		cmake 

 
# 
# Install python packages
#
RUN pip3 install --upgrade pip     
RUN pip3 install --upgrade setuptools  
RUN pip3 install --upgrade numpy scikit-learn matplotlib PyYAML  
RUN pip3 install --upgrade psutil 
RUN pip3 install --upgrade pynvml  
RUN pip3 install --upgrade autopep8
RUN pip3 install --upgrade flake8
RUN pip3 install --upgrade imageio
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 
# add the ROS deb repo to the apt sources list
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		build-essential \
		curl \
		gnupg2 \
		lsb-release \
		ca-certificates
    #&& rm -rf /var/lib/apt/lists/*
    

#
######################
# Install ROS Noetic #
######################
#

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update 
RUN apt-get install ros-noetic-desktop -y --no-install-recommends
RUN apt-get install ros-noetic-tf2* -y --no-install-recommends



# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#           libpython3-dev \
#           python3-rosdep \
#           python3-rosinstall-generator \
#           python3-vcstool \
#           build-essential
	
# RUN apt-get update
# RUN pip3 install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg

# RUN apt-get update && \
#     rosdep init && \
#     rosdep update

# ENV ROS_PKG=ros_base
# ENV ROS_DISTRO=melodic
# ENV ROS_ROOT=/opt/ros/melodic
# RUN pip3 install empy

# RUN mkdir -p /ros_catkin_ws/src && \
# 	cd /ros_catkin_ws && \
# 	catkin config --init -DCMAKE_BUILD_TYPE=Release --blacklist rqt_rviz rviz_plugin_tutorials librviz_tutorial --install

# # setup ROS install
# WORKDIR /ros_catkin_ws
# RUN rosinstall_generator --rosdistro melodic --deps --tar ${ROS_PKG} tf2 > melodic-ros-base.rosinstall

# # Download src
# RUN vcs import --input melodic-ros-base.rosinstall ./src


# # setup environment and install dependencies
# ENV ROS_PYTHON_VERSION=3 

# # # Install wxpython
# # RUN apt-get install -y --no-install-recommends dpkg-dev
	
# # RUN apt-get install -y --no-install-recommends \
# # 	freeglut3-dev  \ 
# # 	libgl1-mesa-dev  \ 
# # 	libglu1-mesa-dev  \ 
# # 	libgstreamer-plugins-base1.0-dev  \ 
# # 	libgtk-3-dev  \ 
# # 	libjpeg-dev  \ 
# # 	libnotify-dev  \ 
# # 	libpng-dev  \ 
# # 	libsdl2-dev  \ 
# # 	libsm-dev  \ 
# # 	libtiff-dev  \ 
# # 	libwebkit2gtk-4.0-dev  \ 
# # 	libxtst-dev  \ 
# # 	doxygen  \ 
# # 	python3-requests

# # RUN pip3 install python-config
# # WORKDIR /wxPython_ws
# # RUN git clone https://github.com/wxWidgets/Phoenix
# # WORKDIR /wxPython_ws/Phoenix
# # RUN git submodule update --init --recursive
# # RUN python3 build.py dox etg --nodoc sip build
# # RUN python3 build.py install

# # Fixing install in ws
# RUN printf "#!/bin/bash\n# Check whether root\nfor pkg in \"\$@\"\ndo\n    echo \"Installing $pkg\"\n    apt-get -my install $pkg >> install.log\ndone\n" > install_skip.sh
# RUN chmod +x install_skip.sh

# # sed magic
# RUN ./install_skip.sh `rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g" | sed -e "s/python/python3/g"`
# RUN rosdep install --from-paths src --ignore-src -y --skip-keys="`rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g"`"
# RUN find . -type f -exec sed -i 's/\/usr\/bin\/env[ ]*python/\/usr\/bin\/env python3/g' {} +


# # RUN catkin build
# #RUN catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m
# RUN python3 ./src/catkin/bin/catkin_make_isolated --install --install-space /opt/ros/melodic -DCMAKE_BUILD_TYPE=Release

# # Reset ros env variables for foxy
# ENV ROS_DISTRO=foxy
# ENV ROS_ROOT=/opt/ros/foxy



#
####################
# Install ROS foxy #
####################
#

RUN apt update && sudo apt install curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update
RUN apt-get install ros-foxy-desktop -y --no-install-recommends








# # 
# # Prepare dependencies
# #
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
		python3-colcon-common-extensions 

# RUN apt-get install -y --no-install-recommends \
# 		libasio-dev \
# 		libtinyxml2-dev \
# 		libcunit1-dev \
# 		libgazebo9-dev \
# 		gazebo9 \
# 		gazebo9-common \
# 		gazebo9-plugin-base

RUN  pip3 install argcomplete \
		flake8-blind-except \
		flake8-builtins \
		flake8-class-newline \
		flake8-comprehensions \
		flake8-deprecated \
		flake8-docstrings \
		flake8-import-order \
		flake8-quotes
	
# #
# # compile yaml-cpp-0.6, which some ROS packages may use (but is not in the 18.04 apt repo)
# #
# RUN git clone --branch yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp yaml-cpp-0.6 && \
#     cd yaml-cpp-0.6 && \
#     mkdir build && \
#     cd build && \
#     cmake -DBUILD_SHARED_LIBS=ON .. && \
#     make -j$(nproc) && \
#     cp libyaml-cpp.so.0.6.0 /usr/lib/x86_64-linux-gnu/ && \
#     ln -s /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.6.0 /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.6

# # 
# # download/build ROS from source
# #
# ARG ROS_PKG=ros_base
# ENV ROS_PKG=ros_base
# # https://answers.ros.org/question/325245/minimal-ros2-installation/?answer=325249#post-id-325249
# RUN mkdir -p ${ROS_ROOT}/src

# RUN cd ${ROS_ROOT} && \
#     rosinstall_generator --deps --tar --rosdistro ${ROS_DISTRO} ${ROS_PKG} \
# 	launch_xml \
# 	launch_yaml \
# 	example_interfaces > ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
#     cat ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall

# #
# # Split the rosinstall file to make sure there in no bug with network
# #
# RUN cd ${ROS_ROOT} && \
# 	split -l 64 ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall seg_ --verbose

# RUN cd ${ROS_ROOT} && \
# 	for f in seg_*; do echo $f && if [[ $(wc -l < $f) -ge 4 ]]; then echo "OK"; fi; done

# #
# # Install each splitted install file
# #
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_aa
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ab
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ac
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ad
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ae
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_af
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ag
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ah
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ai
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_aj
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_ak
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_al
# RUN cd ${ROS_ROOT} && \
# 	vcs import src < seg_am

# #
# # download unreleased packages
# #  
# RUN git clone --branch ros2 https://github.com/Kukanani/vision_msgs ${ROS_ROOT}/src/vision_msgs && \
# 	#git clone --branch debian/foxy/gmock_vendor https://github.com/ros2-gbp/googletest-release ${ROS_ROOT}/src/googletest/gmock_vendor && \
#     git clone --branch ${ROS_DISTRO} https://github.com/ros2/demos /tmp/demos && \
#     cp -r /tmp/demos/demo_nodes_cpp ${ROS_ROOT}/src && \
#     cp -r /tmp/demos/demo_nodes_py ${ROS_ROOT}/src && \
#     rm -r -f /tmp/demos
	
# # patch libyaml - https://github.com/dusty-nv/jetson-containers/issues/41#issuecomment-774767272
# RUN cd ${ROS_ROOT} && \
# 	rm ${ROS_ROOT}/src/libyaml_vendor/CMakeLists.txt && \
# 	wget --no-check-certificate https://raw.githubusercontent.com/ros2/libyaml_vendor/master/CMakeLists.txt -P ${ROS_ROOT}/src/libyaml_vendor/

# #
# # install dependencies using rosdep
# #
# RUN apt-get update && \
#     cd ${ROS_ROOT} && \
#     rosdep update && \
#     rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} -y \
# 		--skip-keys "console_bridge fastcdr fastrtps rti-connext-dds-5.3.1 urdfdom_headers qt_gui ros2interface kdl_parser gmock_vendor"
    
# #
# # build it!
# #
# RUN cd ${ROS_ROOT} && \
# 	colcon build --symlink-install

# #
# # Set the default DDS middleware to cyclonedds
# # https://github.com/ros2/rclcpp/issues/1335
# #
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# REinstall CMake latest version from kitware
RUN apt-get update && \
	apt-get install -y software-properties-common lsb-release
	
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

RUN apt-get update
RUN apt-get install -y --no-install-recommends --upgrade cmake

#
###############
# Final steps #
###############
#

RUN apt-get install ros-noetic-jackal* -y --no-install-recommends
# RUN apt-get install ros-noetic-costmap-converter -y --no-install-recommends
# RUN apt-get install ros-foxy-costmap-converter -y --no-install-recommends

RUN apt-get install xterm -y --no-install-recommends
ENV TERM xterm-256color


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN apt-get install ros-noetic-move-base-flex -y --no-install-recommends
RUN apt-get install libsuitesparse-dev -y --no-install-recommends
RUN apt-get install ros-noetic-libg2o -y --no-install-recommends
RUN apt-get install ros-noetic-pcl-ros -y --no-install-recommends
RUN apt-get install ros-noetic-costmap-converter -y --no-install-recommends
RUN apt-get install python3-catkin-tools -y --no-install-recommends
RUN apt-get install ros-noetic-ros-numpy -y --no-install-recommends

#
# User info
# 
ARG username0=bag
ARG userid0=1085

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda-11.4/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib:


#
# User info
# 
RUN useradd -u $userid0 $username0
RUN adduser $username0 root
RUN adduser $username0 sudo 
RUN echo $username0':myhaljack' | chpasswd  

WORKDIR /home/$username0/eloquent_ws
RUN chown -R $username0:$username0 /home/$username0

USER $username0  

# Set xterm defaults
WORKDIR /home/$username0
RUN echo "xterm*background: black" >> .Xresources
RUN echo "xterm*foreground: lightgray" >> .Xresources
RUN echo "xterm*faceName: monospace:pixelsize=14" >> .Xresources

CMD ["bash"]   
