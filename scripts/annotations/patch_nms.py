with open("generate_annotations.py", "r") as f:
    text = f.read()

search = """        valid_boxes = []
        
        # Keep all boxes with confidence above threshold, but reject "full image" bounding boxes
        # A full image box typically looks roughly like [0.5, 0.5, 1.0, 1.0] in cxcywh
        boxes_list = boxes.cpu().numpy().tolist()"""

replace = """        valid_boxes = []
        
        # Apply Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes
        import torchvision
        boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        nms_idx = torchvision.ops.nms(boxes_xyxy, logits, iou_threshold=0.5)
        
        # Keep all boxes with confidence above threshold, but reject "full image" bounding boxes
        boxes_list = boxes[nms_idx].cpu().numpy().tolist()"""

if search in text:
    text = text.replace(search, replace)
    with open("generate_annotations.py", "w") as f:
        f.write(text)
    print("Patched generate_annotations.py to support NMS")
else:
    print("Search block not found in generate_annotations.py")
