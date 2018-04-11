#!/bin/sh
cd /home/den/Recommender
spark-submit --master local[*] --class recommender.contextualModeling.Predict target/recommender-1.0.0.jar
