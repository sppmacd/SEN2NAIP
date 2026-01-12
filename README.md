Super Resolution of SEN2NAIP dataset (https://huggingface.co/datasets/isp-uv-es/SEN2NAIP) [1]
The dataset consists of 2851 remote sensing imagery data, low and high resolution, in GeoTIFF
format, each of size 484x484 and 4 channels (RGBNIR - Red, Green, Blue, Near Infrared)

I'm going to take only the high-resolution (HR) images, scale them down 2x (to 242x242)
and train a model to upscale them back.

References:

- [1] https://www.nature.com/articles/s41597-024-04214-y
