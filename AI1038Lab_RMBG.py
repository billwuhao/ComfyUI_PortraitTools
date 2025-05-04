# ComfyUI-RMBG v2.2.0
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, BEN, BEN2 and BIREFNET-HR. It leverages deep learning techniques
# to process images and generate masks for background removal.
#
# Models License Notice:
# - RMBG-2.0: Apache-2.0 License (https://huggingface.co/briaai/RMBG-2.0)
# - INSPYRENET: MIT License (https://github.com/plemeri/InSPyReNet)
# - BEN: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN)
# - BEN2: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN2)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-RMBG

import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import folder_paths
from PIL import ImageFilter
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import shutil
import sys
import importlib.util
from transformers import AutoModelForImageSegmentation
import cv2
import types

device = "cuda" if torch.cuda.is_available() else "cpu"

# Add model path
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

# Model configuration
AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "1038lab/RMBG-2.0",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py"
        },
        "cache_dir": "RMBG-2.0"
    },
    "INSPYRENET": {
        "type": "inspyrenet",
        "repo_id": "1038lab/inspyrenet",
        "files": {
            "inspyrenet.safetensors": "inspyrenet.safetensors"
        },
        "cache_dir": "INSPYRENET"
    },
    "BEN": {
        "type": "ben",
        "repo_id": "1038lab/BEN",
        "files": {
            "model.py": "model.py",
            "BEN_Base.pth": "BEN_Base.pth"
        },
        "cache_dir": "BEN"
    },
    "BEN2": {
        "type": "ben2",
        "repo_id": "1038lab/BEN2",
        "files": {
            "BEN2_Base.pth": "BEN2_Base.pth",
            "BEN2.py": "BEN2.py"
        },
        "cache_dir": "BEN2"
    }
}

# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def handle_model_error(message):
    print(f"[RMBG ERROR] {message}")
    raise RuntimeError(message)

class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
    
    def get_cache_dir(self, model_name):
        cache_path = os.path.join(self.base_cache_dir, AVAILABLE_MODELS[model_name]["cache_dir"])
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        
        missing_files = []
        for filename in model_info["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, model_info["files"][filename])):
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache verified"
    
    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            
            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
                    local_dir=cache_dir
                )
                    
            return True, "Model files downloaded successfully"
            
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"
    
    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model = None
        self.current_model_version = None

class RMBGModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            cache_dir = self.get_cache_dir(model_name)
            try:
                # Try standard loading first
                try:
                    self.model = AutoModelForImageSegmentation.from_pretrained(
                        cache_dir,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except AttributeError as ae:
                    if "'Config' object has no attribute 'get_text_config'" in str(ae):
                        print("[RMBG WARNING] Detected newer transformers version, using compatibility mode...")
                        try:
                            from transformers import PreTrainedModel
                            import json
                            
                            config_path = os.path.join(cache_dir, "config.json")
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            
                            birefnet_path = os.path.join(cache_dir, "birefnet.py")
                            BiRefNetConfig_path = os.path.join(cache_dir, "BiRefNet_config.py")
                            
                            # Load the BiRefNetConfig
                            config_spec = importlib.util.spec_from_file_location("BiRefNetConfig", BiRefNetConfig_path)
                            config_module = importlib.util.module_from_spec(config_spec)
                            sys.modules["BiRefNetConfig"] = config_module
                            config_spec.loader.exec_module(config_module)
                            
                            # Fix and load birefnet module
                            with open(birefnet_path, 'r') as f:
                                birefnet_content = f.read()
                            
                            birefnet_content = birefnet_content.replace(
                                "from .BiRefNet_config import BiRefNetConfig", 
                                "from BiRefNetConfig import BiRefNetConfig"
                            )
                            
                            module_name = f"custom_birefnet_model_{hash(birefnet_path)}"
                            module = types.ModuleType(module_name)
                            sys.modules[module_name] = module
                            exec(birefnet_content, module.__dict__)
                            
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr != PreTrainedModel:
                                    BiRefNetConfig = getattr(config_module, "BiRefNetConfig")
                                    model_config = BiRefNetConfig()
                                    self.model = attr(model_config)
                                    
                                    weights_path = os.path.join(cache_dir, "model.safetensors")
                                    try:
                                        try:
                                            import safetensors.torch
                                            self.model.load_state_dict(safetensors.torch.load_file(weights_path))
                                        except ImportError:
                                            from transformers.modeling_utils import load_state_dict
                                            state_dict = load_state_dict(weights_path)
                                            self.model.load_state_dict(state_dict)
                                    except Exception as load_error:
                                        pytorch_weights = os.path.join(cache_dir, "pytorch_model.bin")
                                        if os.path.exists(pytorch_weights):
                                            self.model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
                                        else:
                                            raise RuntimeError(f"Failed to load weights: {str(load_error)}")
                                    break
                            
                            if self.model is None:
                                raise RuntimeError("Could not find suitable model class")
                                
                        except Exception as custom_e:
                            handle_model_error(f"Failed to load model in compatibility mode: {str(custom_e)}")
                    else:
                        raise ae
            except Exception as e:
                handle_model_error(f"Error loading model: {str(e)}")
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
            
    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)

            # Prepare batch processing
            transform_image = transforms.Compose([
                transforms.Resize((params["process_res"], params["process_res"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Ensure input is in list format
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            # Store original image sizes
            original_sizes = [tensor2pil(img).size for img in images]

            # Batch process transformations
            input_tensors = [transform_image(tensor2pil(img)).unsqueeze(0) for img in images]
            input_batch = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                outputs = self.model(input_batch)
                
                if isinstance(outputs, list) and len(outputs) > 0:
                    results = outputs[-1].sigmoid().cpu()
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    results = outputs['logits'].sigmoid().cpu()
                elif isinstance(outputs, torch.Tensor):
                    results = outputs.sigmoid().cpu()
                else:
                    try:
                        if hasattr(outputs, 'last_hidden_state'):
                            results = outputs.last_hidden_state.sigmoid().cpu()
                        else:
                            for k, v in outputs.items():
                                if isinstance(v, torch.Tensor):
                                    results = v.sigmoid().cpu()
                                    break
                    except:
                        handle_model_error("Unable to recognize model output format")
                
                masks = []
                
                # Process each result and resize back to original dimensions
                for i, (result, (orig_w, orig_h)) in enumerate(zip(results, original_sizes)):
                    result = result.squeeze()
                    result = result * (1 + (1 - params["sensitivity"]))
                    result = torch.clamp(result, 0, 1)
                    
                    # Resize back to original dimensions
                    result = F.interpolate(result.unsqueeze(0).unsqueeze(0),
                                         size=(orig_h, orig_w),
                                         mode='bilinear').squeeze()
                    
                    masks.append(tensor2pil(result))

                return masks

        except Exception as e:
            handle_model_error(f"Error in batch processing: {str(e)}")

class InspyrenetModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            try:
                import transparent_background
                self.model = transparent_background.Remover()
                self.current_model_version = model_name
            except ImportError:
                try:
                    import pip
                    pip.main(['install', 'transparent_background'])
                    import transparent_background
                    self.model = transparent_background.Remover()
                    self.current_model_version = model_name
                except Exception as e:
                    handle_model_error(f"Failed to install transparent_background: {str(e)}")
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            # Resize for processing
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            # Process image
            foreground = self.model.process(resized_image, type='rgba')
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in Inspyrenet processing: {str(e)}")

class BENModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            cache_dir = self.get_cache_dir(model_name)
            model_path = os.path.join(cache_dir, "model.py")
            module_name = f"custom_ben_model_{hash(model_path)}"
            
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            ben_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = ben_module
            spec.loader.exec_module(ben_module)
            
            model_weights_path = os.path.join(cache_dir, "BEN_Base.pth")
            self.model = ben_module.BEN_Base()
            self.model.loadcheckpoints(model_weights_path)
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            processed_input = resized_image.convert("RGBA")
            
            with torch.no_grad():
                _, foreground = self.model.inference(processed_input)
            
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in BEN processing: {str(e)}")

class BEN2Model(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            try:
                cache_dir = self.get_cache_dir(model_name)
                model_path = os.path.join(cache_dir, "BEN2.py")
                module_name = f"custom_ben2_model_{hash(model_path)}"
                
                spec = importlib.util.spec_from_file_location(module_name, model_path)
                ben2_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = ben2_module
                spec.loader.exec_module(ben2_module)
                
                model_weights_path = os.path.join(cache_dir, "BEN2_Base.pth")
                self.model = ben2_module.BEN_Base()
                self.model.loadcheckpoints(model_weights_path)
                
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                
                torch.set_float32_matmul_precision('high')
                self.model.to(device)
                self.current_model_version = model_name
                
            except Exception as e:
                handle_model_error(f"Error loading BEN2 model: {str(e)}")
    
    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)
            
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]
            
            batch_size = 3
            all_masks = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_pil_images = []
                original_sizes = []
                
                for img in batch_images:
                    orig_image = tensor2pil(img)
                    w, h = orig_image.size
                    original_sizes.append((w, h))
                    
                    aspect_ratio = h / w
                    new_w = params["process_res"]
                    new_h = int(params["process_res"] * aspect_ratio)
                    resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
                    processed_input = resized_image.convert("RGBA")
                    batch_pil_images.append(processed_input)
                
                with torch.no_grad():
                    try:
                        foregrounds = self.model.inference(batch_pil_images)
                        if not isinstance(foregrounds, list):
                            foregrounds = [foregrounds]
                    except Exception as e:
                        handle_model_error(f"Error in BEN2 inference: {str(e)}")
                
                for foreground, (orig_w, orig_h) in zip(foregrounds, original_sizes):
                    foreground = foreground.resize((orig_w, orig_h), Image.LANCZOS)
                    mask = foreground.split()[-1]
                    all_masks.append(mask)
            
            if len(all_masks) == 1:
                return all_masks[0]
            return all_masks

        except Exception as e:
            handle_model_error(f"Error in BEN2 processing: {str(e)}")

def refine_foreground(image_bchw, masks_b1hw):
    b, c, h, w = image_bchw.shape
    if b != masks_b1hw.shape[0]:
        raise ValueError("images and masks must have the same batch size")
    
    image_np = image_bchw.cpu().numpy()
    mask_np = masks_b1hw.cpu().numpy()
    
    refined_fg = []
    for i in range(b):
        mask = mask_np[i, 0]      
        thresh = 0.45
        mask_binary = (mask > thresh).astype(np.float32)
        
        edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)
        transition_mask = np.logical_and(mask > 0.05, mask < 0.95)
        
        alpha = 0.85
        mask_refined = np.where(transition_mask,
                              alpha * mask + (1-alpha) * edge_blur,
                              mask_binary)
        
        edge_region = np.logical_and(mask > 0.2, mask < 0.8)
        mask_refined = np.where(edge_region,
                              mask_refined * 0.98,
                              mask_refined)
        
        result = []
        for c in range(image_np.shape[1]):
            channel = image_np[i, c]
            refined = channel * mask_refined
            result.append(refined)
            
        refined_fg.append(np.stack(result))
    
    return torch.from_numpy(np.stack(refined_fg))

# class RMBG:
#     def __init__(self):
#         self.models = {
#             "RMBG-2.0": RMBGModel(),
#             "INSPYRENET": InspyrenetModel(),
#             "BEN": BENModel(),
#             "BEN2": BEN2Model()
#         }
    
#     @classmethod
#     def INPUT_TYPES(s):
#         tooltips = {
#             "image": "Input image to be processed for background removal.",
#             "model": "Select the background removal model to use (RMBG-2.0, INSPYRENET, BEN).",
#             "sensitivity": "Adjust the strength of mask detection (higher values result in more aggressive detection).",
#             "process_res": "Set the processing resolution (higher values require more VRAM and may increase processing time).",
#             "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
#             "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
#             "background": "Choose the background color for the final output (Alpha for transparent background).",
#             "invert_output": "Enable to invert both the image and mask output (useful for certain effects).",
#             "optimize": "Enable model optimization for faster processing (may affect output quality).",
#             "refine_foreground": "Use Fast Foreground Colour Estimation to optimize transparent background"
#         }
        
#         return {
#             "required": {
#                 "image": ("IMAGE", {"tooltip": tooltips["image"]}),
#                 "model": (list(AVAILABLE_MODELS.keys()), {"tooltip": tooltips["model"]}),
#             },
#             "optional": {
#                 "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
#                 "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8, "tooltip": tooltips["process_res"]}),
#                 "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
#                 "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
#                 "background": (["Alpha", "black", "white", "gray", "green", "blue", "red"], {"default": "Alpha", "tooltip": tooltips["background"]}),
#                 "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
#                 "optimize": (["default", "on"], {"default": "default", "tooltip": tooltips["optimize"]}),
#                 "refine_foreground": ("BOOLEAN", {"default": False, "tooltip": tooltips["refine_foreground"]})
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
#     RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
#     FUNCTION = "process_image"
#     CATEGORY = "🧪AILab/🧽RMBG"

#     def process_image(self, image, model, **params):
#         try:
#             processed_images = []
#             processed_masks = []
            
#             bg_colors = {
#                 "Alpha": None,
#                 "black": (0, 0, 0),
#                 "white": (255, 255, 255),
#                 "gray": (128, 128, 128),
#                 "green": (0, 255, 0),
#                 "blue": (0, 0, 255),
#                 "red": (255, 0, 0)
#             }
            
#             model_instance = self.models[model]
            
#             # Check and download model if needed
#             cache_status, message = model_instance.check_model_cache(model)
#             if not cache_status:
#                 print(f"Cache check: {message}")
#                 print("Downloading required model files...")
#                 download_status, download_message = model_instance.download_model(model)
#                 if not download_status:
#                     handle_model_error(download_message)
#                 print("Model files downloaded successfully")
            
#             for img in image:
#                 # Get mask from specific model
#                 mask = model_instance.process_image(img, model, params)
                
#                 # Ensure mask is in the correct format
#                 if isinstance(mask, list):
#                     masks = [m.convert("L") for m in mask if isinstance(m, Image.Image)]
#                     mask = masks[0] if masks else None
#                 elif isinstance(mask, Image.Image):
#                     mask = mask.convert("L")

#                 # Post-process mask
#                 mask_tensor = pil2tensor(mask)
#                 mask_tensor = mask_tensor * (1 + (1 - params["sensitivity"]))
#                 mask_tensor = torch.clamp(mask_tensor, 0, 1)
#                 mask = tensor2pil(mask_tensor)
                
#                 if params["mask_blur"] > 0:
#                     mask = mask.filter(ImageFilter.GaussianBlur(radius=params["mask_blur"]))
                
#                 if params["mask_offset"] != 0:
#                     if params["mask_offset"] > 0:
#                         for _ in range(params["mask_offset"]):
#                             mask = mask.filter(ImageFilter.MaxFilter(3))
#                     else:
#                         for _ in range(-params["mask_offset"]):
#                             mask = mask.filter(ImageFilter.MinFilter(3))
                
#                 if params["invert_output"]:
#                     mask = Image.fromarray(255 - np.array(mask))

#                 # Convert to tensors for refine_foreground
#                 img_tensor = torch.from_numpy(np.array(tensor2pil(img))).permute(2, 0, 1).unsqueeze(0) / 255.0
#                 mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255.0

#                 # Create final image
#                 orig_image = tensor2pil(img)
                
#                 if params.get("refine_foreground", False):
#                     refined_fg = refine_foreground(img_tensor, mask_tensor)
#                     refined_fg = tensor2pil(refined_fg[0].permute(1, 2, 0))
#                     r, g, b = refined_fg.split()
#                     foreground = Image.merge('RGBA', (r, g, b, mask))
#                 else:
#                     orig_rgba = orig_image.convert("RGBA")
#                     r, g, b, _ = orig_rgba.split()
#                     foreground = Image.merge('RGBA', (r, g, b, mask))

#                 if params["background"] != "Alpha":
#                     bg_color = bg_colors[params["background"]]
#                     bg_image = Image.new('RGBA', orig_image.size, (*bg_color, 255))
#                     composite_image = Image.alpha_composite(bg_image, foreground)
#                     processed_images.append(pil2tensor(composite_image.convert("RGB")))
#                 else:
#                     processed_images.append(pil2tensor(foreground))
                
#                 processed_masks.append(pil2tensor(mask))

#             # Create mask image for visualization
#             mask_images = []
#             for mask_tensor in processed_masks:
#                 # Convert mask to RGB image format for visualization
#                 mask_image = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
#                 mask_images.append(mask_image)
            
#             mask_image_output = torch.cat(mask_images, dim=0)
            
#             return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0), mask_image_output)
            
#         except Exception as e:
#             handle_model_error(f"Error in image processing: {str(e)}")
#             # Return original image and empty mask on error
#             empty_mask = torch.zeros((image.shape[0], image.shape[2], image.shape[3]))
#             empty_mask_image = empty_mask.reshape((-1, 1, empty_mask.shape[-2], empty_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
#             return (image, empty_mask, empty_mask_image)

# # Node Mapping
# NODE_CLASS_MAPPINGS = {
#     "RMBG": RMBG
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "RMBG": "Remove Background (RMBG)"
# } 