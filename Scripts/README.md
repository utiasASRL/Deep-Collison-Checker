# Deep-Collison-Checker

We provide four scripts here:

- `run_in_container.sh`: runs a command inside the `./SOGM-3D-2D-Net` folder. Without any argument the command is: `python3 train_MyhalCollision.py`. You can choose to execute another command inside the docker container with the argument `-c "your command"`. You can also add the argument -d to run the container in detach mode.

- `dev_noetic.sh`: Starts a container specifically to be used as a development environment. Execute `./dev_noetic.sh -d`, then attach visual studio code to this container named `$USER-noetic_pytorch-dev`.

- `connect_to_container_user.sh`: Connect to a container (starts a new shell in it). Practical for debugging

- `connect_to_container_root.sh`: Connect to a container as root (starts a new shell in it). Practical for debugging