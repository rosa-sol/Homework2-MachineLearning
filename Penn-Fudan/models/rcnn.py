import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_frcnn(num_classes=2, cpu_fast=True, trainable_backbone_layers=0):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        pretrained=True,
        trainable_backbone_layers=trainable_backbone_layers
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if cpu_fast:
        model.rpn.pre_nms_top_n_train = 200
        model.rpn.post_nms_top_n_train = 100
        model.rpn.pre_nms_top_n_test = 100
        model.rpn.post_nms_top_n_test = 50
        model.roi_heads.detections_per_img = 50

    return model
