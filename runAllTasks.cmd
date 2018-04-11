@echo off
rem call spark-submit --master local[*] --class recommender.contextualModeling.PrepareDataSet target/recommender-1.0.0.jar 
call spark-submit --master local[*] --class recommender.contextualModeling.DesignModel target/recommender-1.0.0.jar 
call spark-submit --master local[*] --class recommender.contextualModeling.TrainModel target/recommender-1.0.0.jar 
call spark-submit --master local[*] --class recommender.contextualModeling.Predict target/recommender-1.0.0.jar 
