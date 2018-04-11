#!/usr/bin/env bash
spark-submit --master local[*] --class recommender.contextualModeling.PrepareDataSet target/recommender-1.0.0.jar 
spark-submit --master local[*] --class recommender.contextualModeling.DesignModel target/recommender-1.0.0.jar 
spark-submit --master local[*] --class recommender.contextualModeling.TrainModel target/recommender-1.0.0.jar 
spark-submit --master local[*] --class recommender.contextualModeling.Predict target/recommender-1.0.0.jar 
