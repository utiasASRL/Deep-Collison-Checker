# Deep-Collison-Checker

In this repository, we share the implementation of the paper [Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes](https://arxiv.org/abs/2108.10585). This code is a minimalist version made with two objectives:
 - Easily reproduce the results presented in the paper
 - Possiblity to apply the network to other datasets

 As shown in the next figure, in this repo, we provide the code for our **automated annotation** and **network training**. The whole simulation is ignored, and we provide preprocessed data instead. 

![Intro figure](./Data/github_im.png)


## Introduction

Have a scheme of the original file system and the corresponding docker volumes

Have an instalation guide

 - Set up the docker images
 - Compile the cpp wrappers for KPConv
 - Set up the catkin workspaces

Have some tutorial guides

 - Guide to run the training script alone with preprocessed data that we provide
 - Guide to run the simulator alone and generate data
 - Guide to run the training script on the generated data
 - Guide to show the results of the training
 - Guide to run simulator with a trained model
 - Guide for the development with visual studio: script to forward the tcp ports + visual studio attach + open workspace + Install extension in remote contatiner


