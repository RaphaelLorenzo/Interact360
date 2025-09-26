import numpy as np
import torch

# add the compute euler angles from SixDRep360
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    """Compute the euler angles from the rotation matrices

    Args:
        rotation_matrices (torch.Tensor): Rotation matrices, size [B, 3, 3]

    Returns:
        torch.Tensor: Euler angles, size [B, 3] in radians
    """
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()
        
    x = torch.atan2(R[:,2,1], R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0],R[:,0,0])
    
    xs = torch.atan2(-R[:,1,2], R[:,1,1])
    ys = torch.atan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
        
    # not really needed since we dont use it for training
    # gpu = rotation_matrices.get_device()
    # if gpu < 0:
    #     out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cpu'))
    # else:
    #     out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cuda:%d' % gpu))
    out_euler = torch.zeros(batch,3)
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
        
    return out_euler

# Rotation matrices
def rot_matrix(yaw, pitch, roll):
    """NumPy version for compatibility."""
    yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    Rx = np.array([[1,0,0],
                    [0,np.cos(pitch),-np.sin(pitch)],
                    [0,np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],
                    [0,1,0],
                    [-np.sin(yaw),0,np.cos(yaw)]])
    Rz = np.array([[np.cos(roll),-np.sin(roll),0],
                    [np.sin(roll), np.cos(roll),0],
                    [0,0,1]])
    return Rz @ Ry @ Rx


def rot_matrix_torch(yaw, pitch, roll, device):
    """PyTorch version with GPU support."""
    
    # Convert to radians
    yaw = torch.deg2rad(torch.tensor(yaw, dtype=torch.float32, device=device))
    pitch = torch.deg2rad(torch.tensor(pitch, dtype=torch.float32, device=device))
    roll = torch.deg2rad(torch.tensor(roll, dtype=torch.float32, device=device))
    
    # Create rotation matrices
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
    
    Rx = torch.tensor([[1, 0, 0],
                       [0, cos_pitch, -sin_pitch],
                       [0, sin_pitch, cos_pitch]], dtype=torch.float32, device=device)
    
    Ry = torch.tensor([[cos_yaw, 0, sin_yaw],
                       [0, 1, 0],
                       [-sin_yaw, 0, cos_yaw]], dtype=torch.float32, device=device)
    
    Rz = torch.tensor([[cos_roll, -sin_roll, 0],
                       [sin_roll, cos_roll, 0],
                       [0, 0, 1]], dtype=torch.float32, device=device)
    
    return Rz @ Ry @ Rx


def perspective_view(eq_shape, fov, out_w, out_h, yaw=0, pitch=0, roll=0, z=1):
    """
    Extracts a perspective view from an equirectangular panorama (NumPy version).
    
    Args:
        eq_shape (tuple): Shape of the equirectangular image (h, w).
        fov (float): Horizontal field of view in degrees.
        out_w (int): Output width.
        out_h (int): Output height.
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
        roll (float): Roll angle in degrees.
        z (float): Virtual depth for projection.
    
    Returns:
        map_x, map_y (np.ndarray): Pixel mapping (float32) for remap().
    """
    h, w = eq_shape

    # FOV in radians
    fov_rad = np.deg2rad(fov)
    aspect = out_h / out_w
    fov_y = 2 * np.arctan(np.tan(fov_rad/2) * aspect)

    # Create pixel grid in output image
    x = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), out_w)
    y = np.linspace(-np.tan(fov_y/2), np.tan(fov_y/2), out_h)
    xv, yv = np.meshgrid(x, -y)  # flip y for image coordinates

    # Normalize to unit sphere (z=1 for pinhole projection)
    zv = np.ones_like(xv) * z
    directions = np.stack([xv, yv, zv], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    R = rot_matrix(yaw, pitch, roll)
    dirs_rot = directions @ R.T

    # Convert to spherical coords
    lon = np.arctan2(dirs_rot[...,0], dirs_rot[...,2])
    lat = np.arcsin(dirs_rot[...,1])

    # Map to equirectangular coordinates
    map_x = (lon / (2*np.pi) + 0.5) * w
    map_y = (0.5 - lat / np.pi) * h

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    return map_x, map_y


def perspective_view_torch(eq_shape, fov, out_w, out_h, device, yaw=0, pitch=0, roll=0, z=1):
    """
    PyTorch-accelerated version for GPU computation.
    
    Args:
        eq_shape (tuple): Shape of the equirectangular image (h, w).
        fov (float): Horizontal field of view in degrees.
        out_w (int): Output width.
        out_h (int): Output height.
        device (torch.device): Device to use for computation.
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
        roll (float): Roll angle in degrees.
        z (float): Virtual depth for projection.
    
    Returns:
        map_x, map_y (torch.Tensor): Pixel mapping (float32) for remap(), each of shape (out_h, out_w).
    """

    h, w = eq_shape

    # FOV in radians
    fov_rad = torch.deg2rad(torch.tensor(fov, dtype=torch.float32, device=device))
    aspect = out_h / out_w
    fov_y = 2 * torch.arctan(torch.tan(fov_rad/2) * aspect)

    # Create pixel grid in output image
    x = torch.linspace(-torch.tan(fov_rad/2), torch.tan(fov_rad/2), out_w, device=device)
    y = torch.linspace(-torch.tan(fov_y/2), torch.tan(fov_y/2), out_h, device=device)
    xv, yv = torch.meshgrid(x, -y, indexing='xy')  # flip y for image coordinates

    # Normalize to unit sphere (z=1 for pinhole projection)
    zv = torch.ones_like(xv, device=device) * z
    directions = torch.stack([xv, yv, zv], dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # Get rotation matrix and apply rotation
    R = rot_matrix_torch(yaw, pitch, roll, device)
    dirs_rot = directions @ R.T

    # Convert to spherical coords
    lon = torch.atan2(dirs_rot[..., 0], dirs_rot[..., 2])
    lat = torch.asin(torch.clamp(dirs_rot[..., 1], -1.0, 1.0))  # Clamp to avoid numerical issues

    # Map to equirectangular coordinates
    map_x = (lon / (2 * torch.pi) + 0.5) * w
    map_y = (0.5 - lat / torch.pi) * h

    # # Convert back to NumPy for OpenCV compatibility
    # map_x = map_x.cpu().numpy().astype(np.float32)
    # map_y = map_y.cpu().numpy().astype(np.float32)

    return map_x, map_y

def reverse_perspective_view_torch(eq_shape, fov, out_w, out_h, device, yaw=0, pitch=0, roll=0, z=1):
    """
    Reverse mapping: equirectangular -> perspective projection.

    Args:
        eq_shape (tuple): Shape of the equirectangular image (h, w).
        fov (float): Horizontal field of view in degrees.
        out_w (int): Output width.
        out_h (int): Output height.
        device (torch.device): Device to use for computation.
        yaw, pitch, roll (float): Camera orientation in degrees.
        z (float): Virtual depth.

    Returns:
        map_x, map_y (torch.Tensor): Pixel mapping into perspective image, 
                                     shape (h, w). Invalid pixels set to -1.
    """
    h, w = eq_shape

    # FOV in radians
    fov_rad = torch.deg2rad(torch.tensor(fov, dtype=torch.float32, device=device))
    aspect = out_h / out_w
    fov_y = 2 * torch.arctan(torch.tan(fov_rad/2) * aspect)

    # Build lon/lat grid for the equirectangular image
    xs = torch.linspace(0, w-1, w, device=device)
    ys = torch.linspace(0, h-1, h, device=device)
    xv, yv = torch.meshgrid(xs, ys, indexing='xy')

    lon = (xv / w - 0.5) * 2 * torch.pi
    lat = (0.5 - yv / h) * torch.pi

    # Convert to 3D direction
    dirs = torch.stack([
        torch.sin(lon) * torch.cos(lat),
        torch.sin(lat),
        torch.cos(lon) * torch.cos(lat)
    ], dim=-1)

    # Rotate into camera coords (inverse of what you had)
    R = rot_matrix_torch(yaw, pitch, roll, device)
    dirs_cam = dirs @ R    # instead of dirs @ R.T

    # Project to perspective plane
    xv_proj = dirs_cam[..., 0] / (dirs_cam[..., 2] + 1e-8)
    yv_proj = dirs_cam[..., 1] / (dirs_cam[..., 2] + 1e-8)

    # Scale into [-tan(fov/2), tan(fov/2)]
    x_norm = xv_proj / torch.tan(fov_rad/2)
    y_norm = yv_proj / torch.tan(fov_y/2)

    # Map to pixel coordinates
    map_x = (x_norm + 1) * 0.5 * out_w
    map_y = (1 - (y_norm + 1) * 0.5) * out_h  # flip Y

    # Mask out invalid values (outside FOV or behind camera)
    mask = (dirs_cam[..., 2] > 0) & (torch.abs(x_norm) <= 1) & (torch.abs(y_norm) <= 1)
    map_x = torch.where(mask, map_x, torch.tensor(-1.0, device=device))
    map_y = torch.where(mask, map_y, torch.tensor(-1.0, device=device))

    return map_x, map_y