[中文](README-CN.md)|[English](README.md)

# 人像处理等相关的 ComfyUI 节点

目前包含以下节点:
- 图片人脸对齐(正脸);
- 人脸检测裁剪, 可选是否对齐, 可调裁剪区域大小, 角度;
- 各种证件照一键生成;
- 美化照片, 包括亮度, 饱和度, 锐化, 磨皮等.

示例:

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_07-06-36.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_07-08-46.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_09-05-41.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_09-27-16.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_09-48-23.png)

![](https://github.com/billwuhao/ComfyUI_PortraitTools/blob/main/images/2025-04-11_07-10-24.png)


## 📣 更新

[2025-04-11]⚒️: 发布版本 v1.0.0. 

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_PortraitTools.git
cd ComfyUI_PortraitTools
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

如果你正在使用 [ComfyUI-ReActor](https://github.com/Gourieff/comfyui-reactor) 和 [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) 节点, 不用下载模型, 它们是公用的.

否则, 下载 [detection_Resnet50_Final.pth](https://huggingface.co/salmonrk/facedetection/blob/main/detection_Resnet50_Final.pth) 放到 `ComfyUI\models\facedetection` 文件夹下. [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) 的模型会自动下载到 `ComfyUI\models\RMBG` 文件夹下.

## 鸣谢

感谢以下项目:

- [HivisionIDPhotos](https://github.com/Zeyi-Lin/HivisionIDPhotos)
- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)
- [HivisionIDPhotos-ComfyUI](https://github.com/AIFSH/HivisionIDPhotos-ComfyUI)
- [facerestore_cf](https://github.com/mav-rik/facerestore_cf)
