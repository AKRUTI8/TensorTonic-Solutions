import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    
    Returns:
        List of [x1, y1, x2, y2]
    """
    anchors = []
    
    # Step 1: compute stride
    stride = image_size / feature_size
    
    # Step 2: iterate over grid (row-major: i -> rows, j -> cols)
    for i in range(feature_size):
        for j in range(feature_size):
            
            # Step 3: compute center
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # Step 4: for each scale and aspect ratio
            for s in scales:
                for r in aspect_ratios:
                    
                    # width and height using sqrt formulation
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    # Step 5: compute box corners
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
    
    return anchors