<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="130" height="80"></a>
<a href="https://www.utoronto.ca//"><img src="https://github.com/layer6ai-labs/vl6_recsys2018/blob/master/logos/UofT.jpg" width="120" height="80"></a>
<a href="https://vectorinstitute.ai/"><img src="https://github.com/layer6ai-labs/vl6_recsys2018/blob/master/logos/vector.jpg" width="120" height="80"></a>
</p>

## 2018 ACM RecSys [Challenge](http://www.recsyschallenge.com/2018) 1'st Place Solution For Team vl6 [[paper](http://www.cs.toronto.edu/~mvolkovs/recsys2018_challenge.pdf)]

**Team members**: Maksims Volkovs, Himanshu Rai, Zhaoyue Cheng, Yichao Lu (University of Toronto), Ga Wu (University of Toronto, Vector Institute), Scott Sanner (University of Toronto, Vector Institute)

Contact: maks@layer6.ai

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
//OAuth token for spotify creative api, if doing creative track submission
String authToken = "";

// path to song audio feature file, if doing creative track submission
String creativeTrackFile = "/home/recsys2018/data/song_audio_features.txt";

// path to MPD directory with the JSON files
String trainPath = "/home/recsys2018/data/train/";

// path to challenge set JSON file
String testFile = "/home/recsys2018/data/test/challenge_set.json";

// path to python SVD script included in the repo, default location: script/svd_py.py
String pythonScriptPath = "/home/recsys2018/script/svd_py.py";

//path to cache folder for temp storage, at least 20GB should be available in this folder
String cachePath = "/home/recsys2018/cache/";
```

* Compile and execute with maven:
```
export MAVEN_OPTS="-Xms150g -Xmx150g"
mvn clean compile
mvn exec:java -Dexec.mainClass="main.Executor" 
```
Note that by default the code is executing model for the main track, to run the creative track model set `xgbParams.doCreative = true`. For the creative track we extracted extra song features from the 
[Spotify Audio API](https://developer.spotify.com/documentation/web-api/reference/tracks/get-several-audio-features/). We were able to match most songs from the challenge Million Playlist Dataset, and used the following fields for further feature extraction: `[acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]`. In order to download the data for this track, you need to get the OAuth Token from 
[Spotify API page](https://developer.spotify.com/console/get-audio-features-several-tracks/?ids=4JpKVNYnVcJ8tuMKjAj50A,2NRANZE9UCmPAS5XVbXL40,24JygzOLM0EmRQeGtFcIcG) and
assign it to the `authToken` variable in the `Executor.main` function.

We prioritized speed over memory for this project so you'll need at least 100GB of RAM to run model training and inference. The full end-to-end runtime takes approximately 1.5 days.

