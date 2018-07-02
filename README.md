# ACM RecSys Challenge 2018 Team vl6

**Team members**: Maksims Volkovs, Himanshu Rai, Zhaoyue Cheng, Yichao Lu, Wu Ga

Contact: maks@layer6.ai

<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/recsys2018_vl6/blob/master/logos/layer6ai-logo.png" width="180" height="150"></a>
<a href="https://vectorinstitute.ai/"><img src="https://github.com/layer6ai-labs/recsys2018_vl6/blob/master/logos/vector.jpg" width="180" height="150"></a>
</p>

## Table of Contents  
0. [Introduction](#intro)  
1. [Environment](#env)
2. [Dataset](#dataset)
2. [Executing](#executing)
4. [Results](#results)

<a name="intro"/>

## Introduction
This repository contains the Java implementation of our entries for both main and creative tracks. Our approach 
consists of a two-stage model where in the first stage a blend of collaborative filtering methods is used to 
quickly retrieve a set of candidate songs for each playlist with high recall. Then in the second stage a pairwise 
playlist-song gradient boosting model is used to re-rank the retrieved candidates and maximize precision at the 
top of the recommended list.

<a name="env"/>

## Environment
The model is implemented in Java and tested on the following environment:
* Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* 256GB RAM
* Nvidia Titan V
* Java Oracle 1.8.0_171
* Python, Numpy 1.14.3, Sklearn 0.19.1, Scipy 1.1.0
* Apache Maven 3.3.9
* CUDA 8.0 and CUDNN 8.0
* Intel MKL 2018.1.038
* XGBoost and XGBoost4j 0.7

<a name="dataset"/>



