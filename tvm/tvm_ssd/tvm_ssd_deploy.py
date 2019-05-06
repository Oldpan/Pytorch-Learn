import onnx
import cv2
import tvm
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_runtime
from PIL import Image


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(-scores)

    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box[np.newaxis, :]
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


label_path = 'hand-model-labels.txt'


onnx_model = onnx.load('MobSSD-HandDetect.onnx')

orig_image = cv2.imread('../../datasets/hand-image/hands_2.jpg')
width = orig_image.shape[1]
height = orig_image.shape[0]


# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (300, 300))
# image = np.array(image).transpose((2, 0, 1)).astype('float32') / 255.0
# x = image[np.newaxis, :]


class_names = [name.strip() for name in open(label_path).readlines()]

mean = [127.0, 127.0, 127.0]
std = [128.0]


def transform_image(image):
    image = image - np.array(mean)
    image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image


img = Image.open('../../datasets/hand-image/hands_2.jpg').resize((300, 300))
# img = transform_image(img)

img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img[np.newaxis, :]

img = img/255.0                                                 # remember pytorch tensor is 0-1
x = img


# target = 'llvm'
#
# input_name = '0'
# shape_dict = {input_name: x.shape}
# sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
# #
# with relay.build_config(opt_level=3):
#     graph, lib, params = relay.build_module.build(sym, target, params=params)


print("Reading model files...")
libpath = "MobileNetV2-SSDLite.so"
# lib.export_library(libpath)

graph_json_path = "MobileNetV2-SSDLite.json"
# with open(graph_json_path, 'w') as fo:
#     fo.write(graph)

param_path = "MobileNetV2-SSDLite.params"
# with open(param_path, 'wb') as fo:
#     fo.write(relay.save_param_dict(params))

dtype = 'float32'

loaded_json = open(graph_json_path).read()
loaded_lib = tvm.module.load(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

ctx = tvm.cpu()

m = graph_runtime.create(loaded_json, loaded_lib, ctx)
m.set_input('0', tvm.nd.array(x.astype(dtype)))
m.load_params(loaded_params)

m.run()
scores = m.get_output(0).asnumpy()
boxes = m.get_output(1).asnumpy()

boxes = boxes[0]
scores = scores[0]
prob_threshold = 0.4

picked_box_probs = []
picked_labels = []

for class_index in range(1, scores.shape[1]):
    probs = scores[:, class_index]
    mask = probs > prob_threshold
    probs = probs[mask]
    if probs.shape[0] == 0:
        continue
    subset_boxes = boxes[mask, :]
    box_probs = np.concatenate((subset_boxes, probs.reshape(-1, 1)), axis=1)
    box_probs = hard_nms(box_probs, prob_threshold, top_k=10, candidate_size=200)
    # box_probs = box_utils.nms(box_probs, None,
    #                           score_threshold=prob_threshold,
    #                           iou_threshold=0.45,
    #                           sigma=0.5,
    #                           top_k=10,
    #                           candidate_size=200)
    picked_box_probs.append(box_probs)
    picked_labels.extend([class_index] * box_probs.shape[0])

picked_box_probs = np.concatenate(picked_box_probs)
picked_box_probs[:, 0] *= width
picked_box_probs[:, 1] *= height
picked_box_probs[:, 2] *= width
picked_box_probs[:, 3] *= height

boxes, labels, probs = picked_box_probs[:, :4], np.array(picked_labels), picked_box_probs[:, 4]

for i in range(boxes.shape[0]):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")










