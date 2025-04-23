[中文](README-CN.md) | [English](README.md)

# ComfyUI Nodes for Portrait Processing

Currently includes the following nodes:
- Automatic video watermarking;
- Automatic image watermarking;
- Image face alignment (frontal);
- Face detection and cropping, with optional alignment, adjustable crop area size, and angle;
- One-click generation of various passport photos;
- Photo enhancement, including brightness, saturation, sharpening, and skin smoothing.

Examples:

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-14_21-54-33.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_07-06-36.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_07-08-46.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_09-05-41.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_09-27-16.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_09-48-23.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_07-10-24.png)

## 📣 Updates

[2025-04-24]⚒️: Add automatic video watermarking. 

[2025-04-14]⚒️: Add automatic image watermarking. 

[2025-04-11] ⚒️: Released version v1.0.0.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_PortraitTools.git
cd ComfyUI_PortraitTools
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

If you are using the [ComfyUI-ReActor](https://github.com/Gourieff/comfyui-reactor) and [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) nodes, you do not need to download the models, as they are shared.

Otherwise, download [detection_Resnet50_Final.pth](https://huggingface.co/salmonrk/facedetection/blob/main/detection_Resnet50_Final.pth) and place it in the `ComfyUI\models\facedetection` folder. The models for [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) will be automatically downloaded to the `ComfyUI\models\RMBG` folder.

## Acknowledgements

Thanks to the following projects:

- [HivisionIDPhotos](https://github.com/Zeyi-Lin/HivisionIDPhotos)
- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)
- [HivisionIDPhotos-ComfyUI](https://github.com/AIFSH/HivisionIDPhotos-ComfyUI)
- [facerestore_cf](https://github.com/mav-rik/facerestore_cf)