#!/bin/sh
cd /home/den/Recommender
spark-submit --master local[*] --class recommender.contextualModeling.DesignModel target/recommender-1.0.0.jar
