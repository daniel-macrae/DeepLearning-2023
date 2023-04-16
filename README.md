# DeepLearning-2023
 
In this project for the RUG's Deep Learning course, we apply transfer learning by taking a SSDLite model w. a MobileNetV3 backbone, which has been pretrained on the COCO dataset, and apply it to a combined dataset of football players with the task of detecting players and the ball.

In order to use this code, the two oringinal datasets must be downloaded into a "Datasets" folder in the parent folder of the repository, and then processes by the "Dataset_preprocessing" notebook. The datasets can be downloaded from:
- https://github.com/FootballAnalysis/footballanalysis/tree/main/Dataset/Object%20Detection%20Dataset
- https://universe.roboflow.com/augmented-startups/football-player-detection-kucab
