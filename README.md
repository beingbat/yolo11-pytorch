### `YOLOv11 Code in simple PyTorch`

The layers of the network and building blocks are defined in the `models` folder. It has all the given `weights` by `ultralytics`.

Converted code into this format for more visibility and easier matching and reconstruction with implementation in different frameworks.

Requirements
```
torch                   2.4.1
torchvision             0.19.1
```

How to Run
```
1. In `inference` script, pass parameter to `yolov11` class instance defining model size, possible options `(nano, small, medium, large, xlarge)`. 
2. Load matching weights from `models/weights/` folder, using this you can perform inference on the network with pre-trained weights.
3. Image Preprocess: 
    a. Resize to 640 px and pad to make it square
    b. Normalize image by dividing by 255.
    b. Convert NHWC to NCHW and pass the image(s) to network for inference
```

Due to xlarge weights, using git lfs
You would need to download and install git lfs to pull all files correctly and then run `git lfs install` before cloning repo

Will add more capabilities like train, eval, preprocess soon.
