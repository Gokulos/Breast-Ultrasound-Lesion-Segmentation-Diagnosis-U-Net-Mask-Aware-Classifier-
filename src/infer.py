import numpy as np
import cv2
import tensorflow as tf

from .config import IMG_SIZE, UNET_PATH, CLF_PATH, IDX2CLASS
from .losses import bce_dice_loss, dice_coef

LABELS = [IDX2CLASS[i] for i in range(3)]

def preprocess_single(path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
    return img[None, ..., None]  # (1,H,W,1)

def overlay_mask(gray_img_2d: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    gray = (gray_img_2d * 255).astype(np.uint8)
    mask = (mask_2d > 0.5).astype(np.uint8) * 255
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    out = cv2.addWeighted(gray_bgr, 1.0, mask_bgr, alpha, 0)
    return out

def load_models(unet_path: str = UNET_PATH, clf_path: str = CLF_PATH):
    unet = tf.keras.models.load_model(
        unet_path, custom_objects={"bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef}
    )
    clf = tf.keras.models.load_model(clf_path)
    return unet, clf

def predict_mask_and_class(image_path: str, unet_model, clf_model):
    x = preprocess_single(image_path, IMG_SIZE)

    pred_mask = unet_model.predict(x, verbose=0)[0, :, :, 0]  # (H,W)
    pred_mask_bin = (pred_mask > 0.5).astype(np.float32)

    x2 = np.concatenate([x[0], pred_mask_bin[..., None]], axis=-1)[None, ...]  # (1,H,W,2)
    probs = clf_model.predict(x2, verbose=0)[0]
    cls_id = int(np.argmax(probs))

    return pred_mask, cls_id, probs, LABELS[cls_id]
