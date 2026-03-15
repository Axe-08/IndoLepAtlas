with open("generate_annotations.py", "r") as f:
    text = f.read()

search = """
        # boxes are in cxcywh format, normalized [0,1]
        if len(boxes) == 0:
            return []
        
        # Select the box with highest confidence
        max_idx = logits.argmax().item()
        best_box = boxes[max_idx].cpu().numpy().tolist()
        return [best_box]
"""

replace = """
        # boxes are in cxcywh format, normalized [0,1]
        if len(boxes) == 0:
            return []
            
        valid_boxes = []
        
        # Keep all boxes with confidence above threshold, but reject "full image" bounding boxes
        # A full image box typically looks roughly like [0.5, 0.5, 1.0, 1.0] in cxcywh
        boxes_list = boxes.cpu().numpy().tolist()
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

        return valid_boxes
"""

if search in text:
    text = text.replace(search, replace)
    with open("generate_annotations.py", "w") as f:
        f.write(text)
    print("Patched generate_annotations.py to support multiple bounding boxes")
else:
    print("Search block not found in generate_annotations.py")
