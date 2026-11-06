"""
gradcam.py
──────────
Gradient-weighted Class Activation Mapping (GradCAM) utilities.

Provides:
  - get_gradcam_heatmap  : compute the normalised heatmap array.
  - overlay_gradcam      : blend the heatmap onto the original image.
"""

import numpy as np
import tensorflow as tf
import cv2
from backend.utils.logger import logger


def get_gradcam_heatmap(model, img_array, layer_name: str = "conv2d") -> np.ndarray:
    """
    Compute a GradCAM heatmap for the predicted class.

    Builds a sub-model that exposes both the target convolutional layer's
    output and the final softmax predictions, then uses the gradients of
    the top prediction w.r.t. the conv activations to weight the feature maps.

    Args:
        model      : Loaded tf.keras.Model.
        img_array  : Preprocessed batch array of shape (1, H, W, C).
        layer_name : Name of the convolutional layer to visualise.

    Returns:
        np.ndarray: 2-D normalised heatmap in [0, 1].
    """
    try:
        # Build a model that returns the target layer output + final predictions.
        # model.inputs is already a list — no need to wrap it again.
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            # Explicitly watch the intermediate tensor — GradientTape only
            # auto-watches tf.Variables; without this, gradient() returns None.
            tape.watch(conv_outputs)
            if isinstance(preds, list):
                preds = preds[0]
            # Score for the top predicted class
            loss = preds[:, tf.argmax(preds[0])]

        # Global average pool the gradients over spatial dimensions
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError(
                f"GradCAM: gradient w.r.t. layer '{layer_name}' is None. "
                "Check that the layer name is correct and produces a non-zero activation."
            )
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature maps and collapse to a single 2-D map
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalise to [0, 1]
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    except Exception as e:
        logger.error(f"GradCAM heatmap error: {e}")
        raise


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Blend a GradCAM heatmap onto the original image.

    Uses cv2.addWeighted so both the heatmap and original contribute
    proportionally (heatmap * alpha + original * (1-alpha)), preventing
    the output from becoming over-bright or washed out.

    Args:
        original_img : Decoded BGR image (H, W, 3).
        heatmap      : 2-D float array from get_gradcam_heatmap.
        alpha        : Heatmap intensity weight (default 0.4).

    Returns:
        np.ndarray: Blended BGR image as uint8.
    """
    try:
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # Apply JET colour-map (OpenCV produces BGR output)
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend in BGR space (both images are BGR at this point)
        blended_bgr = cv2.addWeighted(heatmap_colored, alpha, original_img, 1.0 - alpha, 0)

        # Convert to RGB before returning so that callers (PNG encoder, browser)
        # receive correct colours — browsers interpret PNG/JPEG as RGB, not BGR.
        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        logger.error(f"GradCAM overlay error: {e}")
        raise

