# ACM RecSys Challenge 2018 Team vl6

**Team members**: Maksims Volkovs, Himanshu Rai, Zhaoyue Cheng, Yichao Lu, Wu Ga

Contact: maks@layer6.ai

<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/vl6_recsys2018/blob/master/logos/layer6ai-logo.png" width="180" height="150"></a>
<a href="https://vectorinstitute.ai/"><img src="https://github.com/layer6ai-labs/vl6_recsys2018/blob/master/logos/vector.jpg" width="180" height="150"></a>
</p>

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

## Executing

All models are executed from `src/main/java/main/Executor.java`, the main function has examples on 
how to do main and creative track model training, evaluation and submission. To run the model:

* Set all paths:
```
// path to MPD directory with all the JSON files
String trainPath = "/home/recsys2018/data/train/";

// path to challenge set JSON file
String testFile = "/home/recsys2018/data/test/challenge_set.json";

// path to song audio feature file if doing creative track submission
String pythonScriptPath = "/home/recsys2018/data/song_audio_features.txt";

// path to python SVD script included in the repo, default location: script/svd_py.py
String pythonScriptPath = "/home/recsys2018/script/svd_py.py";

//path to cache folder for temp storage
String cachePath = "/home/recsys2018/cache/";
```

* Compile and execute with maven:
```
export MAVEN_OPTS="-Xms150g -Xmx150g"
mvn clean compile
mvn exec:java -Dexec.mainClass="main.Executor" 
```
Note that by default the code is executing model for the main track, to run the creative track model set `xgbParams.doCreative = true`. In order to download the data for the creative track, please get the auth key from [Spotify API page](https://developer.spotify.com/console/get-audio-features-several-tracks/?ids=4JpKVNYnVcJ8tuMKjAj50A,2NRANZE9UCmPAS5XVbXL40,24JygzOLM0EmRQeGtFcIcG) and
assign it to the AUTH_TOKEN variable inside the Executor.downloadCreativeData function. Also note that for this project we prioritized speed over memory so you'll 
need at least 100GB of RAM to train and generate submission.

