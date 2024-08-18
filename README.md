# Visual Object Tracker for 360° Videos

This repository provides the source code of the visual object tracker published in the paper **"Joint Classification and Regression for Visual Tracking with Fully Convolutional Siamese Networks"**, *International Journal of Computer Vision - Springer, 2022*. The main modification I made is adapting this tracker to work on 360° videos, stored in equirectangular format as `.mp4` files.

## Training and Adaptation

This model is currently being trained on an HPC (High Performance Computing) server that uses CPUs. Therefore, you need to adapt this code to use CUDA if you want to train it or run it on a CUDA GPU.

## Modified Scripts

I've reused/modified the following scripts from the PySOT (Python Single Object Tracking) library:

- `pysot/core/config.py`
- `pysot/datasets/augmentation.py`
- `pysot/datasets/dataset.py`
- `pysot/utils/bbox.py`
- `pysot/utils/lr_scheduler.py`

## Reference

URL of the PySOT library: [https://github.com/STVIR/pysot/tree/master/pysot](https://github.com/STVIR/pysot/tree/master/pysot)
