**PLEASE GO TO THE [`report`](https://github.com/sppmacd/SEN2NAIP/tree/report) branch~~~**

# SEN2NAIP Super Resolution

Super Resolution of SEN2NAIP dataset (https://huggingface.co/datasets/isp-uv-es/SEN2NAIP) [1]
The dataset consists of 2851 remote sensing imagery data, low and high resolution, in GeoTIFF
format, each of size 484x484 and 4 channels (RGBNIR - Red, Green, Blue, Near Infrared)

I'm going to take only the high-resolution (HR) images (242x242), scale them down 2 times
(to 121x121) and train a model to upscale them back.

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

## Methods

### Dataset

- https://huggingface.co/datasets/isp-uv-es/SEN2NAIP
- 2851 images
- RGBNIR images; we take only RGB channels to adapt for pre-trained models
- High-resolution and low-resolution images, HR originally 484x484 scaled down to 242x242 because of computational constraints; low-resolution images are 121x121.

### Models

- Bicubic scaling (baseline)
- UNet
- CARN (both "small" and "normal" version tested)

### Loss

- Pixel-wise MSE
- VGG16 content loss

### Training

- Adam/RMSprop optimizer
- For experiments:
  - taking first 256 samples from dataset
  - training stopped early (20 epochs for normal training / 500 or 1000 for overfitting)
  - loss curves show that model has room for improvement

### Tools

- PyTorch via Ignite framework
- TensorBoard for training metrics
- DVC for experiment tracking

## Results

### Sample overfitting

One of the samples was overfitted to check if the model is able to reproduce the image at all.

TODO: Numbers

### Models

All of trained models below bicubic in terms of metrics

TODO: Numbers

### Qualitative results

- Sharper than bicubic
- Common features (e.g roads) are often well represented
- Problems with representing color well
- 2x2 blocks from nearest neighbor upscaling clearly visible in some cases
- Some discolorings (high saturation artifacts)
