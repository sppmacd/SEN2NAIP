# SEN2NAIP Super Resolution

Super Resolution of SEN2NAIP dataset (https://huggingface.co/datasets/isp-uv-es/SEN2NAIP) [1]
The dataset consists of 2851 remote sensing imagery data, low and high resolution, in GeoTIFF
format, each of size 484x484 and 4 channels (RGBNIR - Red, Green, Blue, Near Infrared)

I'm going to take only the high-resolution (HR) images, scale them down 2x (to 242x242)
and train a model to upscale them back.

References:

- [1] https://www.nature.com/articles/s41597-024-04214-y
- https://medium.com/@mickael.boillaud/streamline-your-ml-pipeline-a-comprehensive-guide-to-dvc-data-version-control-97251730b1cf
- DVC example: https://github.com/treeverse/example-get-started/blob/main/src/evaluate.py
- Metrics: https://acta.imeko.org/index.php/acta-imeko/article/download/1679/2939
- U-Net impl: https://github.com/milesial/Pytorch-UNet/
- CARN impl: https://github.com/nmhkahn/CARN-pytorch/
- GAN: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
- Checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
- VGG16 Content loss: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
- ICNR (checkerboard-free pixelshuffle initialization): https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965

UNet_first_run time: ~50min

Table

## Points table

| task                                                                                           | points | done     |
| ---------------------------------------------------------------------------------------------- | ------ | -------- |
| Problem (Super-resolution)                                                                     | 3      | YES      |
| **TOTAL POINTS** (Problem)                                                                     | 3      |          |
| -                                                                                              | -      | -        |
| Model (Ready: U-Net)                                                                           | 1      | YES      |
| Model (Ready: CARN)                                                                            | 1      | YES      |
| Model: GAN                                                                                     | 1      | NO       |
| **TOTAL POINTS** (Model)                                                                       | 2      |          |
| -                                                                                              | -      | -        |
| Training: Training dynamics metrics (Loss + at least 3)                                        | REQ    | YES      |
| Training: Hyperparameter estimation                                                            | 1      | NO       |
| Training: Adaptive hyperparameters                                                             | 1      | NO       |
| Training: Architecture tuning (at least 3 architecture)                                        | 1      | NO       |
| Training: Overfitting some examples from the training set                                      | 1      | YES      |
| Training: Data augmentation                                                                    | 1      | NO       |
| Training: Cross-validation                                                                     | 1      | NO       |
| Training: Distributed learning                                                                 | 1      | NO       |
| Training: Federated learning                                                                   | 2      | NO       |
| Training: Testing various loss functions (at least 3)                                          | 1      | NO (2/3) |
| Training: Calculating intrinsic dimension                                                      | 1      | NO       |
| Training: Custom optimizer                                                                     | 1      | NO       |
| **TOTAL POINTS** (Training)                                                                    | 1      |          |
| -                                                                                              | -      | -        |
| Tools: Git with Readme                                                                         | REQ    | YES      |
| Tools: MLflow,Tensorboard, Neptune, Weights & Biases (along with some analysis of experiments) | 1      | YES      |
| Tools: Run as docker/ docker compose                                                           | 1      | NO       |
| Tools: REST API or GUI for example Gradio, Streamlit                                           | 1      | NO       |
| Tools: DVC                                                                                     | 2      | YES      |
| Tools: Every other MLOps tool                                                                  | 1      | NO       |
| Tools: Label Studio or other data labeling tools                                               | 1      | NO       |
| Tools: Explanation of 3 predictions - e.g. which inputs were most significant                  | 2      | NO       |
| **TOTAL POINTS** (Tools)                                                                       | 3      |          |
| -                                                                                              | -      | -        |
| **TOTAL POINTS** (Problem)                                                                     | 3      |          |
| **TOTAL POINTS** (Model)                                                                       | 2      |          |
| **TOTAL POINTS** (Dataset+Training+Tools+Report)                                               | 4      |          |
| **TOTAL POINTS**                                                                               | 9      |          |

## Report TODO

| task                                                                                     | done |
| ---------------------------------------------------------------------------------------- | ---- |
| Report: description of the data set, with a few image examples                           | NO   |
| Report: description of the problem                                                       | YES  |
| Report: Architectures + Diagrams                                                         | NO   |
| Report: model analysis: size in memory, number of parameters,                            | NO   |
| Report: description of the training and the required commands to run it                  | NO   |
| Report: description of used metrics, loss, and evaluation                                | NO   |
| Report: plots: training and validation loss, metrics                                     | NO   |
| Report: used hyperparameters along with an explanation of each why such value was chosen | NO   |
| Report: comparison of models                                                             | NO   |
| Report: list of libraries and tools used can be a requirements.txt file                  | NO   |
| Report: a description of the runtime environment                                         | NO   |
| Report: training and inference time,                                                     | NO   |
| Report: bibliography                                                                     | NO   |
| Report: A table containing all the completed items with points.                          | NO   |
| Report: Link to Git                                                                      | NO   |
