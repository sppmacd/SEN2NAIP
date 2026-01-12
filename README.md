Super Resolution of SEN2NAIP dataset (https://huggingface.co/datasets/isp-uv-es/SEN2NAIP) [1]
The dataset consists of 2851 remote sensing imagery data, low and high resolution, in GeoTIFF
format, each of size 484x484 and 4 channels (RGBNIR - Red, Green, Blue, Near Infrared)

I'm going to take only the high-resolution (HR) images, scale them down 2x (to 242x242)
and train a model to upscale them back.

References:

- [1] https://www.nature.com/articles/s41597-024-04214-y

https://medium.com/@mickael.boillaud/streamline-your-ml-pipeline-a-comprehensive-guide-to-dvc-data-version-control-97251730b1cf
DVC docs
PyTorch-Ignite docs
DVC example: https://github.com/treeverse/example-get-started/blob/main/src/evaluate.py
Metrics: https://acta.imeko.org/index.php/acta-imeko/article/download/1679/2939
