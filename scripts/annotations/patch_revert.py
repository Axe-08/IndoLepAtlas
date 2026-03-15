with open("generate_annotations.py", "r") as f:
    text = f.read()

search = """        valid_boxes = []
        
        # Apply Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes
        import torchvision
        boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        nms_idx = torchvision.ops.nms(boxes_xyxy, logits, iou_threshold=0.5)
        
        # Keep all boxes with confidence above threshold, but reject "full image" bounding boxes
        boxes_list = boxes[nms_idx].cpu().numpy().tolist()
        for idx in range(len(boxes_list)):
            b = boxes_list[idx]
            cx, cy, w, h = b
            area = w * h
            
            # If the box covers more than 90% of the image, we consider it a generic full-image classification, not a localized box
            if area > 0.90:
                continue
            
            valid_boxes.append(b)
            
        # If all boxes were rejected as full-image bounds, optionally return the best of the rejected (or empty list)
        if not valid_boxes and len(boxes_list) > 0:
            # Fall back to returning the tightest box if everything was large, or just empty list for strictness
            pass

        return valid_boxes"""

replace = """        # Select the box with highest confidence
        max_idx = logits.argmax().item()
        best_box = boxes[max_idx].cpu().numpy().tolist()
        return [best_box]"""

if search in text:
    text = text.replace(search, replace)
    with open("generate_annotations.py", "w") as f:
        f.write(text)
    print("Reverted generate_annotations.py to single highest-confidence box")
else:
    print("Search block not found in generate_annotations.py")

