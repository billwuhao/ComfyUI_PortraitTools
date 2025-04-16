import os
import torch
from copy import deepcopy
import cv2
import numpy as np
from comfy import model_management
import folder_paths
import sys
import math
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from retinaface import RetinaFace
from layout_calculator import generate_layout_photo, generate_layout_image
from beauty import grindSkin, make_whitening, adjust_brightness_contrast_sharpen_saturation
from AILab_RMBG import (AVAILABLE_MODELS,
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

        c = int(math.sqrt(origin_image.size[0] ** 2 + origin_image.size[1] ** 2))
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
        origin_image.paste(
            watermark_mask,
            (int((origin_image.size[0] - c) / 2), int((origin_image.size[1] - c) / 2)),
            mask=watermark_mask.split()[3],
        )
        return origin_image


class ImageWatermark:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "@明文视界"}),
                "font_file": ("STRING", {"default": ""}),
                "angle": ("INT", {"default": 30, "min": 0, "max": 360, "step": 1}),
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "size": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "space": ("INT", {"default": 75, "min": 1, "max": 150, "step": 1}),
                # "chars_per_line": ("INT", {"default": 8, "min": 1, "max": 10, "step": 1}),
                # "font_height_crop": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 5.0, "step": 0.1})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "watermarkgen"
    CATEGORY = "🎤MW/MW-PortraitTools"

    def watermarkgen(
        self,
        image,
        text: str,
        font_file: str,
        angle=30,
        red=0,
        green=0,
        blue=0,
        opacity=0.15,
        size=50,
        space=75,
        chars_per_line=8,
        font_height_crop=1.2,
    ):
        if font_file.strip() == "":
            font_file = os.path.join(current_dir, "ChironGoRoundTC-600SB.ttf")
        watermarker = Watermarker(
            input_image=tensor2pil(image),
            text=text,
            font_file=font_file,
            angle=angle,
            color=f"#{red:02x}{green:02x}{blue:02x}",
            opacity=opacity,
            size=size,
            space=space,
            chars_per_line=chars_per_line,
            font_height_crop=font_height_crop,
        )

        return (pil2tensor(watermarker.image),)

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

class IDPhotos:
    def __init__(self):
        self.models = {
            "RMBG-2.0": RMBGModel(),
            "INSPYRENET": InspyrenetModel(),
            "BEN": BENModel(),
            "BEN2": BEN2Model()
        }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
                "rmbg_model":(list(AVAILABLE_MODELS),{"default":"RMBG-2.0"}),
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
        
    def gen_img(self, image, rmbg_model, bg_color, size, face_reduction, face_up_down, angle_offset, kb, dpi=300):
        # 解析尺寸参数
        size_parts = size.split(',')
        size = (int(size_parts[1]), int(size_parts[2])) 

        hd_photo = self.photo_gen(image, size, face_reduction, face_up_down, angle_offset, by_size=False)
        rmbg_hd_photo = self.image_rmbg(hd_photo, rmbg_model, bg_color)
        standard_photo = self.photo_gen(image, size, face_reduction, face_up_down, angle_offset, by_size=True)
        rmbg_standard_photo = self.image_rmbg(standard_photo, rmbg_model, bg_color)
        print_photos = self.print_photos_gen(rmbg_standard_photo, size, kb, dpi)

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
        model_instance = self.models[model]
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
    "DetectCropFace": DetectCropFaces,
    "AlignFace": AlignFace,
    "IDPhotos": IDPhotos,
    "BeautifyPhoto": BeautifyPhoto,
    "ImageWatermark": ImageWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetectCropFaces": "Detect and Crop Faces",
    "AlignFace": "Align Face",
    "IDPhotos": "ID Photos",
    "BeautifyPhoto": "Beautify Photo",
    "ImageWatermark": "Image Watermark",
}