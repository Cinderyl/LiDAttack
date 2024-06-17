import torch
import cv2
import numpy as np

def visualize_gradcam(net, input_dict, class_idx):
    # get feature map and gradient
    feature_map = net.backbone(input_dict['points'])
    gradient = net.get_gradient(feature_map)

    # get class activation map
    cam = net.get_cam(feature_map, gradient, class_idx)

    # resize feature map and gradient
    feature_map = feature_map.squeeze().cpu().numpy()
    gradient = gradient.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (feature_map.shape[1], feature_map.shape[0]))

    # normalize feature map and gradient
    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
    gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient))

    # apply heatmap to feature map
    heatmap = cv2.applyColorMap(np.uint8(gradient * 255), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(feature_map)
    cam = cam / np.max(cam)

    return feature_map, gradient, cam
