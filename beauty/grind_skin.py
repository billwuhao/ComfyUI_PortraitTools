# Required Libraries
import cv2


def grindSkin(src, grindDegree: int = 3, detailDegree: int = 1, strength: int = 9):
    """
    Dest =(Src * (100 - Opacity) + (Src + 2 * GaussBlur(EPFFilter(Src) - Src)) * Opacity) / 100
    人像磨皮方案
    Args:
        src: 原图
        grindDegree: 磨皮程度调节参数
        detailDegree: 细节程度调节参数
        strength: 融合程度，作为磨皮强度（0 - 10）

    Returns:
        磨皮后的图像
    """
    if strength <= 0:
        return src
    dst = src.copy()
    opacity = min(10.0, strength) / 10.0
    dx = grindDegree * 5
    fc = grindDegree * 12.5
    temp1 = cv2.bilateralFilter(src[:, :, :3], dx, fc, fc)
    temp2 = cv2.subtract(temp1, src[:, :, :3])
    temp3 = cv2.GaussianBlur(temp2, (2 * detailDegree - 1, 2 * detailDegree - 1), 0)
    temp4 = cv2.add(cv2.add(temp3, temp3), src[:, :, :3])
    dst[:, :, :3] = cv2.addWeighted(temp4, opacity, src[:, :, :3], 1 - opacity, 0.0)
    return dst

