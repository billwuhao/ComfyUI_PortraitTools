import os
import torch
from copy import deepcopy
import cv2
import numpy as np
from comfy import model_management
import folder_paths
import sys
import math
from typing import List, Optional, Union
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops, ImageOps, ImageSequence

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .retinaface import RetinaFace
from .layout_calculator import generate_layout_photo, generate_layout_image
from .beauty import grindSkin, make_whitening, adjust_brightness_contrast_sharpen_saturation
from .AI1038Lab_RMBG import (AVAILABLE_MODELS,
                        RMBGModel,
                        BENModel,
                        BEN2Model,
                        InspyrenetModel,
                        tensor2pil,
                        pil2tensor,
                        handle_model_error,
)

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "facedetection", "detection_Resnet50_Final.pth")
device = model_management.get_torch_device()


def init_model(half=False, device=device):
    model = RetinaFace(network_name='resnet50', device=device, half=half)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)

    return model


def tensor_to_rgb(tensor_image):
    """
    将ComfyUI的tensor图像转换为RGB图像
    
    参数:
        tensor_image: 形状为[B,H,W,C]的tensor，通常为float32类型，值范围0-1
        
    返回:
        numpy数组，RGB格式，uint8类型，值范围0-255
    """
    # ComfyUI的tensor格式是[B,H,W,C]，取第一张图片
    if len(tensor_image.shape) == 4:
        image = tensor_image[0].cpu().numpy()
    else:
        image = tensor_image.cpu().numpy()
    
    # 转换为0-255范围的uint8
    image = (image * 255.0).astype(np.uint8)
    
    return image


def rgb_to_tensor(image):
    """
    将RGB图像转换回ComfyUI的tensor格式
    
    参数:
        image: numpy数组，RGB格式，uint8类型
        
    返回:
        形状为[1,H,W,C]的tensor，float32类型，值范围0-1
    """
    # 转换为float32并归一化到0-1
    image = image.astype(np.float32) / 255.0
    
    # 转换为tensor并添加批次维度
    image = torch.from_numpy(image).unsqueeze(0)
    
    return image


class Watermarker(object):
    """图片水印工具"""

    def __init__(
        self,
        input_image: Image.Image,
        text: str,
        font_file: str,
        angle=30,
        color="#8B8B1B",
        opacity=0.15,
        size=50,
        space=75,
        chars_per_line=8,
        font_height_crop=1.2,
        offset_x=0,
        offset_y=0,
    ):
        """_summary_

        Parameters
        ----------
        input_image : Image.Image
            PIL图片对象
        text : str
            水印文字
        angle : int, optional
            水印角度, by default 30
        color : str, optional
            水印颜色, by default "#8B8B1B"
        font_file : str, optional
            字体文件, by default "青鸟华光简琥珀.ttf"
        font_height_crop : float, optional
            字体高度裁剪比例, by default 1.2
        opacity : float, optional
            水印透明度, by default 0.15
        size : int, optional
            字体大小, by default 50
        space : int, optional
            水印间距, by default 75
        chars_per_line : int, optional
            每行字符数, by default 8
        """
        self.input_image = input_image
        self.text = text
        self.angle = angle
        self.color = color
        self.font_file = font_file
        self.font_height_crop = font_height_crop
        self.opacity = opacity
        self.size = size
        self.space = space
        self.chars_per_line = chars_per_line
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.image = self._add_mark_striped()

    @staticmethod
    def set_image_opacity(image, opacity: float):
        alpha = image.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        image.putalpha(alpha)
        return image

    @staticmethod
    def crop_image_edge(image):
        bg = Image.new(mode="RGBA", size=image.size)
        diff = ImageChops.difference(image, bg)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image

    def _add_mark_striped(self):
        origin_image = self.input_image.convert("RGBA")
        width = len(self.text) * self.size
        height = round(self.size * self.font_height_crop)
        watermark_image = Image.new(mode="RGBA", size=(width, height))
        draw_table = ImageDraw.Draw(watermark_image)
        draw_table.text(
            (0, 0),
            self.text,
            fill=self.color,
            font=ImageFont.truetype(self.font_file, size=self.size),
        )
        watermark_image = Watermarker.crop_image_edge(watermark_image)
        Watermarker.set_image_opacity(watermark_image, self.opacity)

        # 确保水印覆盖整个图像，增加水印掩码的尺寸
        c = int(math.sqrt(origin_image.size[0] ** 2 + origin_image.size[1] ** 2) * 1.5)
        watermark_mask = Image.new(mode="RGBA", size=(c, c))
        
        y, idx = 0, 0
        while y < c:
            x = -int((watermark_image.size[0] + self.space) * 0.5 * idx)
            idx = (idx + 1) % 2
            while x < c:
                watermark_mask.paste(watermark_image, (x, y))
                x += watermark_image.size[0] + self.space
            y += watermark_image.size[1] + self.space

        watermark_mask = watermark_mask.rotate(self.angle)
        # 计算安全的偏移范围
        max_offset = c // 4  # 限制最大偏移量为水印掩码尺寸的1/4
        safe_offset_x = max(min(self.offset_x, max_offset), -max_offset)
        safe_offset_y = max(min(self.offset_y, max_offset), -max_offset)
        
        paste_x = int((origin_image.size[0] - c) / 2) + safe_offset_x
        paste_y = int((origin_image.size[1] - c) / 2) + safe_offset_y
        origin_image.paste(
            watermark_mask,
            (paste_x, paste_y),
            mask=watermark_mask.split()[3],
        )

        return origin_image


input_dir = folder_paths.get_input_directory()

def get_path():
    from pathlib import Path
    import yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(script_dir, "extra_help_file.yaml")
    try:
        # 尝试打开并加载 YAML 文件
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            images_dir = data["images_dir"]
            images_dir = Path(images_dir)
            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Customize images loading path not found: {images_dir}")

            print(f"Customize images loading path: {images_dir}")
            return images_dir
    except FileNotFoundError:
        print(f"Error: File not found - extra_help_file.yaml")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except KeyError:
        print(f"Error: Missing key 'images_dir' in YAML file.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # 如果加载失败，返回默认路径
    print("No customize images loading path found, use default path.")
    return input_dir

def get_all_files(
    root_dir: str,
    return_type: str = "list",
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    relative_path: bool = False
) -> Union[List[str], dict]:
    """
    递归获取目录下所有文件路径
    
    :param root_dir: 要遍历的根目录
    :param return_type: 返回类型 - "list"(列表) 或 "dict"(按目录分组)
    :param extensions: 可选的文件扩展名过滤列表 (如 ['.py', '.txt'])
    :param exclude_dirs: 要排除的目录名列表 (如 ['__pycache__', '.git'])
    :param relative_path: 是否返回相对路径 (相对于root_dir)
    :return: 文件路径列表或字典
    """
    file_paths = []
    file_dict = {}
    
    # 规范化目录路径
    root_dir = os.path.normpath(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 处理排除目录
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_files = []
        for filename in filenames:
            # 扩展名过滤
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            # 构建完整路径
            full_path = os.path.join(dirpath, filename)
            
            # 处理相对路径
            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)
            
            current_files.append(full_path)
        
        if return_type == "dict":
            # 使用相对路径或绝对路径作为键
            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)
    
    return file_dict if return_type == "dict" else file_paths


class LoadImageMW:
    images_dir = get_path()
    files = get_all_files(images_dir, return_type="list", extensions=[".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"], relative_path=True)
    for i in files:
        import shutil
        src_path = folder_paths.get_annotated_filepath(i, images_dir)
        dst_path = os.path.join(input_dir, i)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"image": (sorted(s.files), {"image_upload": True})},}

    CATEGORY = "🎤MW/MW-PortraitTools"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        import node_helpers
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        import hashlib
        image_path = folder_paths.get_annotated_filepath(image, s.images_dir)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    # @classmethod
    # def VALIDATE_INPUTS(s, image):
    #     if not folder_paths.exists_annotated_filepath(image + "[input]"):
    #         return "Invalid image file: {}".format(image)

    #     return True


class ImageWatermark:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "@明文视界"}),
                "font_file": ("STRING", {"default": ""}),
                "angle": ("INT", {"default": 30, "min": 0, "max": 360, "step": 1}),
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "size": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "space": ("INT", {"default": 75, "min": 1, "max": 150, "step": 1}),
                "movement_type": (["None", "up_down", "left_right", "angel_change"], {"default": "None"}),
                "movement_amount": ("FLOAT", {"default": 1, "min": 0.2, "max": 5, "step": 0.2}),
                # "chars_per_line": ("INT", {"default": 8, "min": 1, "max": 10, "step": 1}),
                # "font_height_crop": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 5.0, "step": 0.1})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "watermarkgen"
    CATEGORY = "🎤MW/MW-PortraitTools"

    def watermarkgen(
        self,
        images,
        text: str,
        font_file: str,
        angle=30,
        red=0,
        green=0,
        blue=0,
        opacity=0.15,
        size=50,
        space=75,
        movement_type="None",
        movement_amount=1,
        chars_per_line=8,
        font_height_crop=1.2,
    ):
        if font_file.strip() == "":
            font_file = os.path.join(current_dir, "ChironGoRoundTC-600SB.ttf")
            
        # 处理批量图像
        batch_size = images.shape[0]
        result_tensors = []
        
        for i in range(batch_size):
            # 获取当前图像
            current_image = images[i:i+1]
            pil_image = tensor2pil(current_image)
            
            # 获取图像尺寸
            img_width, img_height = pil_image.size
            
            # 计算安全的最大偏移量（不超过图像尺寸的1/4）
            max_offset_x = img_width // 8
            max_offset_y = img_height // 8
            
            # 根据移动类型计算当前图像的水印参数
            current_angle = angle
            current_offset_x = 0
            current_offset_y = 0
            
            if movement_type == "angel_change":
                # 角度在原始角度的基础上变化
                current_angle = (angle + i * movement_amount) % 360
            elif movement_type == "up_down":
                # 上下移动水印位置，使用正弦函数实现平滑循环
                cycle_position = (i * movement_amount) % (max_offset_y * 4)
                current_offset_y = int(math.sin(cycle_position * math.pi / (max_offset_y * 2)) * max_offset_y)
            elif movement_type == "left_right":
                # 左右移动水印位置，使用正弦函数实现平滑循环
                cycle_position = (i * movement_amount) % (max_offset_x * 4)
                current_offset_x = int(math.sin(cycle_position * math.pi / (max_offset_x * 2)) * max_offset_x)
            # elif movement_type == "circular":
            #     # 圆形轨迹移动
            #     cycle = 2 * math.pi * (i * movement_amount % 100) / 100
            #     current_offset_x = int(math.cos(cycle) * max_offset_x)
            #     current_offset_y = int(math.sin(cycle) * max_offset_y)
            else:
                # 不移动水印位置
                current_offset_x = 0
                current_offset_y = 0

            watermarker = Watermarker(
                input_image=pil_image,
                text=text,
                font_file=font_file,
                angle=current_angle,
                color=f"#{red:02x}{green:02x}{blue:02x}",
                opacity=opacity,
                size=size,
                space=space,
                chars_per_line=chars_per_line,
                font_height_crop=font_height_crop,
                offset_x=current_offset_x,
                offset_y=current_offset_y
            )

            # 转换回tensor并添加到结果列表
            result_tensor = pil2tensor(watermarker.image)
            result_tensors.append(result_tensor)
        
        # 合并所有结果为一个批量tensor
        return (torch.cat(result_tensors, dim=0),)

class AlignFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "half":  ("BOOLEAN", {"default": False}),
                "angle_offset": ("FLOAT", {"default": 1.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                # "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detect_and_align_whole_image"
    CATEGORY = "🎤MW/MW-PortraitTools"

    def detect_and_align_whole_image(self,
                                image, 
                                half, 
                                angle_offset=1.0,
                                conf_threshold=0.8,
                                nms_threshold=0.4, 
                                use_origin_size=True):
        # 初始化模型
        face_detector = init_model(half=half, device=device)
        
        # 读取图像
        img_rgb = tensor_to_rgb(image)
        
        # 检测人脸
        face_info = face_detector.detect_faces(img_rgb, 
                                conf_threshold=conf_threshold, 
                                nms_threshold=nms_threshold, 
                                use_origin_size=use_origin_size
                                )

        if len(face_info) == 0:
            print("未检测到人脸")
            return (image,)
        
        # 获取最大的人脸（假设主要人脸是最大的）
        areas = (face_info[:, 2] - face_info[:, 0]) * (face_info[:, 3] - face_info[:, 1])
        max_face_idx = np.argmax(areas)
        face_box = face_info[max_face_idx, 0:4]
        landmarks = face_info[max_face_idx, 5:15].reshape(5, 2)
        
        # 提取关键点
        facial5points = [[landmarks[j][0], landmarks[j][1]] for j in range(5)]
        
        # 使用简单的方法进行对齐
        # 计算眼睛中心点（使用前两个关键点，它们通常是左右眼）
        left_eye = facial5points[0]
        right_eye = facial5points[1]
        
        # 计算眼睛之间的角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 计算眼睛中心
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(eye_center, angle + angle_offset, 1)
        
        # 对整个图像进行旋转
        h, w = img_rgb.shape[:2]
        aligned_image = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        image_tensor = rgb_to_tensor(aligned_image)

        # 重新检测旋转后的人脸，以获取新的边界框
        rotated_face_info = face_detector.detect_faces(aligned_image, 
                                conf_threshold=conf_threshold, 
                                nms_threshold=nms_threshold, 
                                use_origin_size=use_origin_size
                                )
        
        if len(rotated_face_info) == 0:
            print("对齐后未检测到人脸，返回原始图像")
            return (image,)
        else:  
            return (image_tensor,)


class DetectCropFaces:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "half":  ("BOOLEAN", {"default": False}),
                "horizontal_padding": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "vertical_padding": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "do_align": ("BOOLEAN", {"default": True}),
                "angle_offset": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                # "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detect_and_align_faces"
    CATEGORY = "🎤MW/MW-PortraitTools"

    def detect_and_align_faces(self,
                                image, 
                                half, 
                                horizontal_padding,
                                vertical_padding,
                                do_align=True, 
                                angle_offset=0.1,
                                conf_threshold=0.8,
                                nms_threshold=0.4, 
                                use_origin_size=True):
        # 初始化模型
        face_detector = init_model(half=half, device=device)
        
        # 读取图像
        img_rgb = tensor_to_rgb(image)

        _, aligned_faces = face_detector.align_multi(
                                img_rgb, 
                                padding=(horizontal_padding, vertical_padding), 
                                do_align=do_align, 
                                angle_offset=angle_offset,
                                conf_threshold=conf_threshold, 
                                nms_threshold=nms_threshold, 
                                use_origin_size=use_origin_size,
                                limit=None)

        if len(aligned_faces) == 0:
            print("没有检测到人脸，返回原始图像")
            return (image,)

        # 如果只检测到一个人脸，直接返回
        if len(aligned_faces) == 1:
            face_tensor = rgb_to_tensor(aligned_faces[0])
            return (face_tensor,)
        
        face_tensors = []
        # 找出所有人脸中最大的尺寸
        max_h = max([face.shape[0] for face in aligned_faces])
        max_w = max([face.shape[1] for face in aligned_faces])
        
        for i, face in enumerate(aligned_faces):
            # 调整所有人脸到相同大小
            resized_face = cv2.resize(face, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
            face_tensor = rgb_to_tensor(resized_face)
            face_tensors.append(face_tensor)

        # 将所有人脸tensor拼接成一个批次
        batch_tensor = torch.cat(face_tensors, dim=0)
        
        # 返回批次tensor
        return (batch_tensor,)


size_list = [
    "一寸,413,295",
    "二寸,626,413",
    "小一寸,378,260",
    "小二寸,531,413",
    "大一寸,567,390",
    "大二寸,626,413",
    "五寸,1499,1050",
    "教师资格证,413,295",
    "国家公务员考试,413,295",
    "初级会计考试,413,295",
    "英语四六级考试,192,144",
    "计算机等级考试,567,390",
    "研究生考试,709,531",
    "社保卡,441,358",
    "电子驾驶证,378,260",
    "美国签证,600,600",
    "日本签证,413,295",
    "韩国签证,531,413" 
]

bg_colors = {
    "Alpha": None,
    "black": (0, 0, 0),             
    "white": (255, 255, 255),      
    "gray": (128, 128, 128),      
    "green": (0, 255, 0),         
    "pure_blue": (0, 0, 255),    
    "pure_red": (255, 0, 0),      
    "cornflower_blue": (98, 139, 206),  
    "crimson_red": (215, 69, 50),      
    "dark_slate_blue": (75, 97, 144),  
    "snow_white": (242, 240, 240)  
}

MODEL_CACHE = None
class IDPhotos:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
                "rmbg_model":(list(AVAILABLE_MODELS),{"default":"RMBG-2.0"}),
                "unload_model":("BOOLEAN",{"default":True}),
                "bg_color":(list(bg_colors.keys()),{"default":"Alpha"}),
                "size":(size_list,{"default":"一寸,413,295"}),
                "kb":("INT",{"default":500,"min":5,"max":2000,"step":1}),
                "dpi":("INT",{"default":300,"min":50,"max":1000,"step":10}),
                "face_reduction":("FLOAT",{
                    "default": 1.0,
                    "min":0.0,
                    "max":5.0,
                    "step":0.1,
                }),
                "face_up_down":("FLOAT",{
                    "default": 0.0,
                    "min":-0.5,
                    "max":0.5,
                    "step":0.01,
                }),
                "angle_offset":("FLOAT",{
                    "default": 1.0,
                    "min":-10.0,
                    "max":10.0,
                    "step":0.1,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("standard_photo", "hd_photo", "print_photos")
    FUNCTION = "gen_img"
    CATEGORY = "🎤MW/MW-PortraitTools"
        
    def gen_img(self, image, rmbg_model, unload_model, bg_color, size, face_reduction, face_up_down, angle_offset, kb, dpi=300):
        # 解析尺寸参数
        size_parts = size.split(',')
        size = (int(size_parts[1]), int(size_parts[2])) 

        hd_photo = self.photo_gen(image, size, face_reduction, face_up_down, angle_offset, by_size=False)
        rmbg_hd_photo = self.image_rmbg(hd_photo, rmbg_model, bg_color)
        standard_photo = self.photo_gen(image, size, face_reduction, face_up_down, angle_offset, by_size=True)
        rmbg_standard_photo = self.image_rmbg(standard_photo, rmbg_model, bg_color)
        print_photos = self.print_photos_gen(rmbg_standard_photo, size, kb, dpi)
        
        if unload_model:
            global MODEL_CACHE
            MODEL_CACHE = None
            torch.cuda.empty_cache()

        return (rmbg_standard_photo, rmbg_hd_photo, print_photos)
    
    def photo_gen(self, image, size, face_reduction, face_up_down, angle_offset, by_size=True):
        # 初始化模型
        face_detector = init_model(half=True, device=device)
        
        # 读取图像
        img_rgb = tensor_to_rgb(image)

        _, aligned_faces = face_detector.align_multi(
                                img_rgb, 
                                angle_offset=angle_offset,
                                padding=(face_reduction, face_reduction), 
                                do_align=True, 
                                conf_threshold=0.8, 
                                nms_threshold=0.4, 
                                use_origin_size=True,
                                limit=None)

        if len(aligned_faces) > 1:
            raise ValueError("Multiple faces detected, please upload an image of a single face.")
        if len(aligned_faces) == 0:
            raise ValueError("No face detected, please upload an image containing the face.")

        face = aligned_faces[0]
        
        # 获取目标尺寸
        target_h = size[0]
        target_w = size[1]
        
        # 获取人脸图像的尺寸
        face_h, face_w = face.shape[:2]
        
        if by_size:
            # 模式1: 按指定尺寸调整
            # 计算缩放比例，使照片高度比目标高度多 target_h * face_up_down
            scale_h = (target_h + target_h * abs(face_up_down)) / face_h
            scaled_w = int(face_w * scale_h)
            scaled_h = int(face_h * scale_h)
            # 如果宽度小于目标宽度，继续放大
            if scaled_w < target_w:
                scale_w = target_w / scaled_w
                scaled_w = target_w
                scaled_h = int(scaled_h * scale_w)
            
            # 缩放图像
            scaled_face = cv2.resize(face, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 计算裁剪区域
            center_x = scaled_w // 2
            # 移动裁剪区域
            center_y = int(scaled_h // 2 + target_h * face_up_down / 2)
            
            # 计算裁剪区域的左上角和右下角坐标
            left = center_x - target_w // 2
            right = left + target_w
            top = center_y - target_h // 2
            bottom = top + target_h
            
            # 确保裁剪区域在图像内
            if left < 0:
                left = 0
                right = target_w
            if right > scaled_w:
                right = scaled_w
                left = scaled_w - target_w
            if top < 0:
                top = 0
                bottom = target_h
            if bottom > scaled_h:
                bottom = scaled_h
                top = scaled_h - target_h
            
            # 裁剪图像
            final_image = scaled_face[top:bottom, left:right]
            
            # 确保最终图像尺寸正确
            if final_image.shape[:2] != (target_h, target_w):
                final_image = cv2.resize(final_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            # 模式2: 严格按比例调整尺寸，不缩放照片
            # 计算缩放比例，使照片高度比目标高度多 target_h * face_up_down
            adjusted_target_h = int(face_h/(1 + abs(face_up_down)))
            adjusted_target_w = int(target_w * (adjusted_target_h / target_h))
            
            # 如果调整后的宽度超过了照片宽度，需要重新计算
            if adjusted_target_w > face_w:
                # 按照宽度计算比例
                scale_w = face_w / target_w
                adjusted_target_w = face_w
                adjusted_target_h = int(target_h * scale_w)
            
            # 计算裁剪区域
            center_x = face_w // 2
            # 移动裁剪区域
            center_y = int(face_h // 2 + adjusted_target_h * face_up_down / 2)
            
            # 计算裁剪区域的左上角和右下角坐标
            left = center_x - adjusted_target_w // 2
            right = left + adjusted_target_w
            top = center_y - adjusted_target_h // 2
            bottom = top + adjusted_target_h
            
            # 确保裁剪区域在图像内
            if left < 0:
                left = 0
                right = adjusted_target_w
            if right > face_w:
                right = face_w
                left = face_w - adjusted_target_w
            if top < 0:
                top = 0
                bottom = adjusted_target_h
            if bottom > face_h:
                bottom = face_h
                top = face_h - adjusted_target_h
            
            # 裁剪图像
            final_image = face[top:bottom, left:right]

        # 转换回tensor格式
        result_tensor = rgb_to_tensor(final_image)
        
        return result_tensor

    def print_photos_gen(self, input_image, size, kb, dpi=300):
        # 将tensor转换为RGB图像
        img_rgb = tensor_to_rgb(input_image)

        from io import BytesIO
        pil_img = Image.fromarray(img_rgb)
        
        # 创建字节流对象
        img_byte_arr = BytesIO()
        
        # 保存到字节流
        pil_img.save(img_byte_arr, format="PNG", dpi=(dpi, dpi))
        img_byte_arr.seek(0)

        # 调整图像大小到指定KB
        quality = 95
        while True:
            # 创建字节流对象
            img_byte_arr = BytesIO()
            
            # 保存图像到字节流
            pil_img.save(img_byte_arr, format="PNG", quality=quality, dpi=(dpi, dpi))
            
            # 获取图像大小(KB)
            img_size_kb = len(img_byte_arr.getvalue()) / 1024
            
            # 检查图像大小是否在目标范围内
            if img_size_kb <= kb or quality == 1:
                # 如果图像小于目标大小，添加填充
                if img_size_kb < kb:
                    padding_size = int(
                        (kb * 1024) - len(img_byte_arr.getvalue())
                    )
                    padding = b"\x00" * padding_size
                    img_byte_arr.write(padding)
                
                break
            
            # 如果图像仍然太大，降低质量
            quality -= 5
            
            # 确保质量不低于1
            if quality < 1:
                quality = 1
        
        # 将字节流转换回PIL图像
        img_byte_arr.seek(0)
        pil_img = Image.open(img_byte_arr)
        
        result_layout_photo = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2RGB)
        
        # 生成布局
        typography_arr, typography_rotate = generate_layout_photo(
            input_height=size[0], input_width=size[1]
        )
        
        # 生成最终布局图像
        result_layout_image = generate_layout_image(
            result_layout_photo,
            typography_arr,
            typography_rotate,
            height=size[0],
            width=size[1],
        )
        
        # 转换为RGB并转换为tensor
        print_cv2 = cv2.cvtColor(result_layout_image, cv2.COLOR_BGR2RGB)
        print_photos = rgb_to_tensor(print_cv2)
        
        return print_photos

    def image_rmbg(self, image, model, bg_color):
        global MODEL_CACHE
        if MODEL_CACHE is None:
            MODEL_CACHE = {
                "RMBG-2.0": RMBGModel(),
                "INSPYRENET": InspyrenetModel(),
                "BEN": BENModel(),
                "BEN2": BEN2Model()
            }

        model_instance = MODEL_CACHE[model]
        params = {
            "sensitivity": 1.0,
            "process_res": 1024,
            "mask_blur": 0,
            "mask_offset": 0,
            "background": bg_colors[bg_color],
            "invert_output": False,
            "optimize": "default",
            "refine_foreground": False
        }
        # Check and download model if needed
        cache_status, message = model_instance.check_model_cache(model)
        if not cache_status:
            print(f"Cache check: {message}")
            print("Downloading required model files...")
            download_status, download_message = model_instance.download_model(model)
            if not download_status:
                handle_model_error(download_message)
            print("Model files downloaded successfully")
        
        # Get mask from specific model
        mask = model_instance.process_image(image, model, params)

        # Ensure mask is in the correct format
        if isinstance(mask, list):
            masks = [m.convert("L") for m in mask if isinstance(m, Image.Image)]
            mask = masks[0] if masks else None
        elif isinstance(mask, Image.Image):
            mask = mask.convert("L")

        # Post-process mask
        mask_tensor = pil2tensor(mask)
        mask_tensor = mask_tensor * (1 + (1 - params["sensitivity"]))
        mask_tensor = torch.clamp(mask_tensor, 0, 1)
        mask = tensor2pil(mask_tensor)
        
        # Create final image
        orig_image = tensor2pil(image)

        orig_rgba = orig_image.convert("RGBA")
        r, g, b, _ = orig_rgba.split()
        foreground = Image.merge('RGBA', (r, g, b, mask))

        if bg_color != "Alpha":
            bg_color = bg_colors[bg_color]
            bg_image = Image.new('RGBA', orig_image.size, (*bg_color, 255))
            composite_image = Image.alpha_composite(bg_image, foreground)
            processed_image = pil2tensor(composite_image.convert("RGB"))
        else:
            processed_image = pil2tensor(foreground)
        
        return processed_image
        

class BeautifyPhoto:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
                "whitening_strength":("INT",{
                    "default": 0,
                    "min":0,
                    "max":100,
                    "step":1,
                }),
                 "brightness_strength":("INT",{
                    "default": 0,
                    "min":-100,
                    "max":100,
                    "step":1,
                }),
                "contrast_strength":("INT",{
                    "default": 0,
                    "min":-100,
                    "max":100,
                    "step":1,
                }),
                "saturation_strength":("INT",{
                    "default": 0,
                    "min":-100,
                    "max":100,
                    "step":1,
                }),
                "sharpen_strength":("FLOAT",{
                    "default": 0.1,
                    "min":0.0,
                    "max":10.0,
                    "step":0.1,
                }),
                "grind_skin":("INT",{
                    "default": 0,
                    "min":0,
                    "max":10,
                    "step":1,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "beautify"
    CATEGORY = "🎤MW/MW-PortraitTools"
        
    def beautify(self,
                image,
                whitening_strength,
                brightness_strength,
                contrast_strength,
                saturation_strength,
                sharpen_strength,
                grind_skin,
                ):
        
        img_np = tensor_to_rgb(image)
        input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        
        adjusted_image = grindSkin(input_image, strength=grind_skin)
        adjusted_image = make_whitening(adjusted_image, strength=whitening_strength)
        adjusted_image = adjust_brightness_contrast_sharpen_saturation(
            adjusted_image, 
            brightness_factor=brightness_strength,
            contrast_factor=contrast_strength,
            sharpen_strength=sharpen_strength,
            saturation_factor=saturation_strength,
            )

        result_image = cv2.cvtColor(adjusted_image,cv2.COLOR_BGR2RGB)
        result_tensor = rgb_to_tensor(result_image)

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "LoadImageMW": LoadImageMW,
    "DetectCropFace": DetectCropFaces,
    "AlignFace": AlignFace,
    "IDPhotos": IDPhotos,
    "BeautifyPhoto": BeautifyPhoto,
    "ImageWatermark": ImageWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageMW": "Load Image @MW",
    "DetectCropFaces": "Detect and Crop Faces",
    "AlignFace": "Align Face",
    "IDPhotos": "ID Photos",
    "BeautifyPhoto": "Beautify Photo",
    "ImageWatermark": "Image Watermark",
}