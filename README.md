# Deep-Collison-Checker

In this repository, we share the implementation of the paper [Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes](https://arxiv.org/abs/2108.10585). This code is a minimalist version made with two objectives:
 - Easily reproduce the results presented in the paper
 - Possiblity to apply the network to other datasets

 As shown in the next figure, in this repo, we provide the code for our **automated annotation** and **network training**. The whole simulation is ignored, and we provide preprocessed data instead. 

![Intro figure](./Data/github_im.png)


## Setup

For convenience we provide a Dockerfile which builds a docker image able to run the code. Please refer to [./Docker/README.md](./Docker) for detailed setup instructions

## Data

### Preprocessed data for fast reproducable results

We provide preprocessed data coming from our simulator. Simply download this [zip file]() and extract its content in the `./Data` folder. 

You should end up with the folder `./Data/Simulation/simulated_runs`, containing 20 dated folders. The first one contains the mapping session of the environment. The rest are sessions performed among Bouncers.

In the folder `./Data/Simulation/slam_offline`, we provide the preprocessed results of the mapping session. A *.ply* file containing the pointmap of the environment.

Eventually the folder `./Data/Simulation/calibration` contains the lidar extrinsec calibration.


### Instructions to apply on a different dataset

If you want to use our network on your own data, the first simple solution is to reproduce the exact same format for your own data. 

1) modify the calibration file according to your own lidar calibration. 
2) Create a file `./Data/Simulation/slam_offline/YYYY-MM-DD-HH-MM-SS/map_update_0001.ply`. See our [pointmap creation code](SOGM-3D-2D-Net/datasets/MyhalCollision.py#L1724) for how to create such a map. 
3) Organise every data folder in `./Data/Simulation/simulated_runs` as follows:

```
    #   YYYY-MM-DD-HH-MM-SS
    #   |
    #   |---logs-YYYY-MM-DD-HH-MM-SS
    #   |   |
    #   |   |---map_traj.ply         # (OPTIONAL) map_poses = loc_poses (x, y, z, qx qy, qz, qw)
    #   |   |---pointmap_00000.ply   # (OPTIONAL) pointmap of this particular session
    #   |
    #   |---sim_frames
    #   |   |
    #   |   |---XXXX.XXXX.ply  # .ply file for each frame point cloud (x, y, z) in lidar corrdinates.
    #   |   |---XXXX.XXXX.ply
    #   |   |--- ...
    #   |
    #   |---gt_pose.ply  # (OPTIONAL) groundtruth poses (x, y, z, qx, qy, qz, qw)
    #   |
    #   |---loc_pose.ply  # localization poses (x, y, z, qx, qy, qz, qw)
```

4) You will have to modify the paths ans parameters according to your new data [here](SOGM-3D-2D-Net/train_MyhalCollision.py#L283-L303). Also choose which folder you use as validation [here](SOGM-3D-2D-Net/train_MyhalCollision.py#L312)

5) There might be errors along the way, try to follow the code execution and correct the possible mistakes


## Run the Annotation and Network

We provide a script to run the code inside a docker container using the image in the `./Docker` folder. Simply start it with:

```
cd Scripts
./run_in_container.sh
```

This script runs a command inside the `./SOGM-3D-2D-Net` folder. Without any argument the command is: `python3 train_MyhalCollision.py`. This script first annotated the data and creates preprocessed 2D+T point clouds in `./Data/Simulation/collisions`. Then it starts the training on this data, generating SOGMs from the preprocessed 2D+T clouds

You can choose to execute another command inside the docker container with the argument `-c`. For example, you can plot the current state of your network training with:

```
cd Scripts
./run_in_container.sh -c "python3 collider_plots.py"
```

You can also add the argument -d to run the container in detach mode (very practical as the training lasts several hours).


## Building a dev environment using Docker and VSCode

We provide a simple way to develop over our code using Docker and VSCode. First start a docker container specifically for development:

```
cd Scripts
./dev_noetic.sh -d
```

Then then attach visual studio code to this container named `$USER-noetic_pytorch-dev`. For this you need to install the docker extension, then go to the list of docker containers running, right click on `$USER-noetic_pytorch-dev`, and `attach visual studio code`.

You can even do it over shh by forwarding the right port. Execute the following commands (On windows, it can be done using MobaXterm local terminal):

```
set DOCKER_HOST="tcp://localhost:23751"
ssh -i "path_to_your_ssh_key" -NL localhost:23751:/var/run/docker.sock  user@your_domain_or_ip
```

The list of docker running on your remote server should appear in the list of your local VSCode. YOu will probably need the extensions `Remote-SSH` and `Remote-Containers`.


## Going further

If you are interested in using this code with our simulator, you can go to the two following repositories:

- [https://github.com/utiasASRL/MyhalSimulator](https://github.com/utiasASRL/MyhalSimulator)

- [https://github.com/utiasASRL/MyhalSimulator-DeepCollider](https://github.com/utiasASRL/MyhalSimulator-DeepCollider)

The first one contains the code to run our Gazebo simulations, and the second on contains the code to perform online predictions within the simulator. Note though, that these repositories are still in developpement and are not as easily run as this one.


## Reference

If you are to use this code, please cite our paper

```
@article{thomas2021learning,
    Author = {Thomas, Hugues and Gallet de Saint Aurin, Matthieu and Zhang, Jian and Barfoot, Timothy D.},
    Title = {Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes},
    Journal = {arXiv preprint arXiv:2108.10585},
    Year = {2021}
}
```

## License
Our code is released under MIT License (see LICENSE file for details).