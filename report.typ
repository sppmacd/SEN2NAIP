= SEN2NAIP Super Resolution

Super Resolution of SEN2NAIP dataset (https://huggingface.co/datasets/isp-uv-es/SEN2NAIP)
The dataset consists of 2851 remote sensing imagery data, low and high resolution, in GeoTIFF
format, each of size 484x484 and 4 channels (RGBNIR - Red, Green, Blue, Near Infrared)

I'm going to take only the high-resolution (HR) images (242x242), scale them down 2 times
(to 121x121) and train a model to upscale them back.


== Dataset

- https://huggingface.co/datasets/isp-uv-es/SEN2NAIP
- 2851 images
- RGBNIR images; we take only RGB channels to adapt for pre-trained models
- High-resolution and low-resolution images, HR originally 484x484 scaled down to 242x242 because of computational constraints; low-resolution images are 121x121.

#figure(
  grid(
    columns: (1fr, 1fr),
    image("images/dataset1.png"), image("images/dataset2.png"),
    image("images/dataset3.png"), image("images/dataset4.png"),
  ),
  caption: [Dataset examples (RGB)],
)

== Models

Used bicubic scaling as a baseline.

=== UNet

#image("images/unet.drawio.png")

```
================================================================
Total params: 31,038,404
Trainable params: 31,038,404
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.89
Forward/backward pass size (MB): 3658.23
Params size (MB): 118.40
Estimated Total Size (MB): 3777.53
----------------------------------------------------------------
```

Standard U-Net, except using LeakyReLU instead of ReLU.

=== CARN

#figure(
  grid(
    image("images/carn1.png"),
    image("images/carn2.png"),
  ),
  caption: [CARN architecture (images taken from paper)],
)

#figure(
  image("images/carn_heads.png"),
  caption: [Modified CARN head and upsample block],
)

==== Summary

```
================================================================
Total params: 838,267
Trainable params: 838,243
Non-trainable params: 24
----------------------------------------------------------------
Input size (MB): 0.17
Forward/backward pass size (MB): 706.63
Params size (MB): 3.20
Estimated Total Size (MB): 709.99
---------------------------------------------------------------
```

Both "small" (`CARN_m`) and "normal" version was tested.

The final model is `CARN` with some modifications:
- `LeakyReLU` instead of `ReLU` everywhere
- Reflect padding instead of zeros everywhere
- More conv layers in the head
- Replaced upsample block with nearest-neighbor upscaling + leaky ReLU - this effectively removes all checkerboard artifacts

== Loss

- Pixel-wise MSE (L2)
- Pixel-wise MAE (L1)
- *VGG16 content loss*

== Training

- *Adam* / RMSprop optimizer
- For experiments:
  - taking first 256 samples from dataset
  - training stopped early (20 epochs for normal training / 500 or 1000 for overfitting)
  - loss curves show that model has room for improvement
- Trained for ~52 minutes.
- To run full pipeline run `dvc repro` in the repo root.
- Hyperparameters:
  - batch size: 16 (resource constraints, dataset size)
  - max epochs: 20 (resource constraints)
  - learning rate: 1e-3 (highest with reasonably stable training)

== Tools

- PyTorch via Ignite framework
- TensorBoard for training metrics
- DVC for experiment tracking

=== TensorBoard analysis

#figure(
  grid(
    columns: 3,
    image("images/Screenshot_20260126_145042.png"),
    image("images/Screenshot_20260126_145057.png"),
    image("images/Screenshot_20260126_145101.png"),

    image("images/Screenshot_20260126_145104.png"), image("images/Screenshot_20260126_145111.png"),
  ),
  caption: [TensorBoard plots for last (`final`) training],
)

- Showing 3 metrics: grad norm sparsity, gradient L2, weights L2
- weights L2 goes upwards throughout the entire training (likely because of model converging to the average output, while starting from 0)
- loss curves show that model has room for improvement (still going downwards)
- grad L2 spikes are correlated with loss spikes/instabilities

== Results

=== Sample overfitting

One of the samples was overfitted to check if the model is able to reproduce the image at all.

#figure(
  table(
    columns: (3cm, 2cm, 2cm),
    align: (left, right, right),
    [model], [test SSIM], [test PSNR],
    [final], [0.92], [36.33],
  ),
  caption: [Comparison of overfitted sample to bicubic baseline],
)

#image("images/overfitting.png")

The images and metrics show that the model generally is capable of representing the final image, although the generated image is slightly more blurred than the original.

=== Models

Metrics: *SSIM*, *PSNR*

All of trained models are below bicubic in terms of metrics.

#figure(
  table(
    columns: (3cm, 2cm, 2cm),
    align: (left, right, right),
    [model], [test SSIM], [test PSNR],
    [bicubic], [0.90], [35.15],
    [final], [0.88], [33.05],
  ),
  caption: [Comparison of final model to the bicubic baseline (quantitative)],
)

=== Qualitative results

- Sharper than bicubic
- Common features (e.g roads) are often well represented
- Problems with representing color well, especially when undertrained
- Some discolorings (high saturation artifacts/RGB noise on upscaled areas)

#show figure: set block(breakable: true)
#figure(
  grid(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      [Ground truth], [Low-resolution], [Model], [Bicubic],
    ),
    image("images/results1.png"),
    image("images/results2.png"),
    image("images/results3.png"),
    image("images/results4.png"),
    image("images/results5.png"),
  ),
  caption: [Model performance compared to ground truth and bicubic upscaling],
)

=== Performance

Average inference time: 85.70 ms/img (tested on 64 batches 4 images each)

== Testing setup

- OS: EndeavourOS Linux x86_64 (kernel: 6.18.5-arch1-1)
- Host: 82K2 IdeaPad Gaming 3 15ACH6
- CPU: AMD Ryzen 5 5600H with Radeon Graphics (12) @ 3.301GHz
- GPU: NVIDIA GeForce GTX 1650 Mobile / Max-Q (4GB VRAM)
- RAM: 32 GB
- Disk: SSD

#pagebreak()

== Points table

#table(
  columns: (1fr, 2cm),
  stroke: none,
  table.header([task], [points]),
  table.hline(),
  [Problem (Super-resolution)], [3      ],

  table.hline(),
  [*TOTAL POINTS* (Problem)], [3      ],
  table.hline(),

  [Model (Ready: U-Net)], [1      ],

  [Model (Ready: CARN)], [1      ],

  table.hline(),
  [*TOTAL POINTS* (Model)], [2      ],
  table.hline(),

  [Training: Overfitting some examples from the training set], [1      ],

  [Training: Data augmentation], [1      ],

  [Training: Testing various loss functions (at least 3)], [1      ],

  table.hline(),
  [*TOTAL POINTS* (Training)], [3      ],
  table.hline(),

  [Tools: MLflow,Tensorboard, Neptune, Weights & Biases (along with some analysis of experiments) ], [1      ],

  [Tools: DVC], [2      ],

  table.hline(),
  [*TOTAL POINTS* (Tools)], [3      ],
  table.hline(),
  [*TOTAL POINTS* (Problem)], [3      ],

  [*TOTAL POINTS* (Model)], [2      ],

  [*TOTAL POINTS* (Dataset+Training+Tools+Report)], [6      ],
  table.hline(),

  [*TOTAL POINTS*   ], [ 11     ],
  table.hline(),
)

#pagebreak()

== References

- DVC: https://medium.com/@mickael.boillaud/streamline-your-ml-pipeline-a-comprehensive-guide-to-dvc-data-version-control-97251730b1cf
- DVC example: https://github.com/treeverse/example-get-started/blob/main/src/evaluate.py
- Metrics: https://acta.imeko.org/index.php/acta-imeko/article/download/1679/2939
- U-Net impl: https://github.com/milesial/Pytorch-UNet/
- CARN: https://arxiv.org/pdf/1803.08664
- CARN impl: https://github.com/nmhkahn/CARN-pytorch/
- GAN: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
- Checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
- VGG16 Content loss: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
- ICNR (checkerboard-free pixelshuffle initialization): https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
- Zero-order hold for checkerboard-artifact-free CNNs: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F3A72B4581D101881B4A08C09150914/S2048770319000027a.pdf/checkerboard-artifacts-free-convolutional-neural-networks.pdf

#pagebreak()

#set page(flipped: true)

== Experiments list (from DVC)

Notes:
- Experiments are sorted chronologically (later experiments are first)
- SSIM was incorrectly calculated for max=255 instead of max=1; displaying $-log_10(1-"SSIM")$ instead (higher is better)
- Effective `batch_size` is `gradient_accumulation_steps` #sym.times `batch_size`.
- `train_time` is in minutes

#let results = csv("experiments-t.csv")

#table(
  columns: 11,
  ..results.flatten()
)
