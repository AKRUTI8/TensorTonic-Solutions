import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    
    Args:
        T: (4,4) transformation matrix
        points: (3,) or (N,3)
    
    Returns:
        Transformed points: (3,) or (N,3)
    """
    points = np.asarray(points)
    
    # Check if single point
    single = False
    if points.ndim == 1:
        points = points.reshape(1, 3)
        single = True
    
    # Convert to homogeneous coordinates (append 1)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  # (N,4)
    
    # Apply transformation
    transformed_h = (T @ points_h.T).T  # (N,4)
    
    # Drop homogeneous coordinate
    transformed = transformed_h[:, :3]
    
    # Return in original shape
    if single:
        return transformed[0]
    return transformed