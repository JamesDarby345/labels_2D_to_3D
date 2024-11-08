import os
import numpy as np
import argparse
import cv2
from pathlib import Path
import open3d as o3d
from ppm import Ppm
from typing import Optional, Dict
import zarr
import dask.array as da
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

def depth_overlay_2d_to_3d_zarr(zarr_path, ppm_path, ppm_mask_path, overlay_folder_path, surf_val=32, zarr_size=(256,256,256), zarr_chunks=(128,128,128), z_range=None, radius=1, dir=1):
    try:
        # Check if required files/folders exist
        if not os.path.exists(ppm_path):
            raise FileNotFoundError(f"PPM file not found: {ppm_path}")
        if not os.path.exists(ppm_mask_path):
            raise FileNotFoundError(f"PPM mask file not found: {ppm_mask_path}")
        if not os.path.exists(overlay_folder_path):
            raise FileNotFoundError(f"Overlay folder not found: {overlay_folder_path}")

        # Create zarr if it doesn't exist
        if not os.path.exists(zarr_path):
            zarr.create(
                shape=zarr_size,
                chunks=zarr_chunks,
                dtype=np.uint8,
                store=zarr_path,
                fill_value=0
            )
        
        # Load zarr
        z = zarr.open(zarr_path, mode='r+')
        if isinstance(z, zarr.hierarchy.Group):
            z = z[0]
        # data = da.from_zarr(z, chunks=(256,256,256))
        
        # Load PPM and mask
        ppm = Ppm.loadPpm(Path(ppm_path))
        ppm.loadData()
        ppm_mask = cv2.imread(ppm_mask_path, cv2.IMREAD_GRAYSCALE)
        overlay_manager = DataSliceManager(overlay_folder_path)
        overlay_manager.z_range = z_range
        
        # print(f"Found {len(overlay_manager.z_values)} overlay files")
        
        # Process each overlay
        for z_value in overlay_manager.z_values:
            overlay = overlay_manager.get_data(z_value)
            if overlay is None:
                continue
                
            # print(f"Processing overlay for z-value: {z_value}")
            
            # Add these debug prints right after loading the overlay
            # print(f"Initial overlay unique values: {np.unique(overlay)}")
            # print(f"Initial overlay > 0 count: {np.count_nonzero(overlay > 0)}")

            # Check the initial rows/cols
            rows, cols = np.where(overlay > 0)
            # print(f"Initial rows/cols count: {len(rows)}")
            
            # Filter using ppm_mask
            if ppm_mask is not None:
                # print(f"PPM mask shape: {ppm_mask.shape}")
                # print(f"PPM mask unique values: {np.unique(ppm_mask)}")
                mask = ppm_mask[rows, cols] > 0
                # print(f"Points remaining after ppm_mask: {np.count_nonzero(mask)}")
                rows = rows[mask]
                cols = cols[mask]
                
            # print(f"Overlay shape: {overlay.shape}")
            # print(rows.shape, cols.shape)
            # Get 3D points
            coords = np.stack([rows, cols], axis=-1).astype(np.float64)
            xyz_points = ppm.ijk_interpolator(coords)
            
            # Apply offset from normal
            if z_value != surf_val:
                normals = ppm.normal_interpolator(coords)
                xyz_points += dir * normals * (z_value - surf_val)
            
            # Convert to voxel coordinates and filter out-of-bounds points
            voxel_coords = xyz_points.astype(np.int32)
            # print(f"Voxel coord ranges before bounds check:")
            # print(f"X: {voxel_coords[:,0].min()}-{voxel_coords[:,0].max()}")
            # print(f"Y: {voxel_coords[:,1].min()}-{voxel_coords[:,1].max()}")
            # print(f"Z: {voxel_coords[:,2].min()}-{voxel_coords[:,2].max()}")
            # print(f"Voxel coords shape: {voxel_coords.shape}")
            # print(f"Zarr size: {zarr_size}")

            # zarr coords are z,y,x voxels coords are xyz here...
            mask = (
                (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < zarr_size[0]) &
                (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < zarr_size[1]) &
                (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < zarr_size[2])
            )
            # print(f"Points remaining after bounds check: {np.count_nonzero(mask)}")
            valid_coords = voxel_coords[mask]
            valid_values = overlay[rows[mask], cols[mask]]
            # print("valid_coords.shape, valid_values.shape", valid_coords.shape, valid_values.shape)
            
            # print(valid_coords[:, 2].max(), valid_coords[:, 1].max(), valid_coords[:, 0].max())
            # print(valid_coords[:, 2].min(), valid_coords[:, 1].min(), valid_coords[:, 0].min())
            # print(np.unique(valid_values))
            
            # Set initial voxels
            z[valid_coords[:, 2], valid_coords[:, 1], valid_coords[:, 0]] = valid_values
            
            neighbors = get_neighbors(radius)
            for dx, dy, dz in neighbors:
                neighbor_coords = valid_coords + np.array([dx, dy, dz])
                #zarr zyx coords, voxel coords xyz here...
                neighbor_mask = (
                    (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < zarr_size[0]) &
                    (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < zarr_size[1]) &
                    (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < zarr_size[2])
                )
                
                # Only set neighbor voxels that are within bounds and currently 0
                valid_neighbors = neighbor_coords[neighbor_mask]
                if len(valid_neighbors) > 0:
                    current_values = z[valid_neighbors[:, 2], valid_neighbors[:, 1], valid_neighbors[:, 0]]
                    zero_mask = current_values == 0
                    if np.any(zero_mask):
                        z[valid_neighbors[zero_mask, 2], 
                        valid_neighbors[zero_mask, 1], 
                        valid_neighbors[zero_mask, 0]] = valid_values[neighbor_mask][zero_mask]
        
        print("Finished processing overlays")
        
    except Exception as e:
        error_msg = f"Error processing segment: {str(e)}"
        # If called directly, raise the error
        if __name__ == "__main__":
            raise Exception(error_msg)
        # If called from multi_segment_folder, just print and return
        print(error_msg)
        return

def get_neighbors(radius):
    """Returns array of 3D neighbor offsets within given radius"""
    if radius <= 0:
        return []
        
    neighbors = []
    r = int(radius)
    for x in range(-r, r+1):
        for y in range(-r, r+1):
            for z in range(-r, r+1):
                # Skip center point
                if x == 0 and y == 0 and z == 0:
                    continue
                    
                # Check if point is within radius
                if (x*x + y*y + z*z) <= radius*radius:
                    neighbors.append((x, y, z))
    # print(neighbors)
    return neighbors

def create_point_cloud(ppm, overlay, ppm_mask=None, color=None, z_diff=0, layer=None):
    """Creates point cloud from overlay coordinates using PPM data"""
    # Get coordinates where overlay > 0
    rows, cols = np.where(overlay > 0)
    
    # Filter coordinates using ppm_mask if provided
    if ppm_mask is not None:
        mask = ppm_mask[rows, cols] > 0
        rows = rows[mask]
        cols = cols[mask]

    # Stack coordinates for interpolation
    coords = np.stack([rows, cols], axis=-1).astype(np.float64)
    
    # Get 3D points from PPM
    xyz_points = ppm.ijk_interpolator(coords)
    
    # If z_diff is provided, offset points along normals
    if z_diff:
        normals = ppm.normal_interpolator(coords)
        xyz_points += normals * z_diff
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    
    # Use TIF slice values for colors if provided, otherwise use uniform color
    if layer is not None:
        print("Applying data layer coloring")
        # Get color values from tif_slice at the overlay coordinates
        colors = layer[rows, cols].astype(float) / 255.0  # Normalize to [0,1]
        colors = np.stack([colors, colors, colors], axis=-1)  # Convert to RGB
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif color is not None:
        pcd.paint_uniform_color(color)
    
    return pcd

class DataSliceManager:
    def __init__(self, data_folder_path: str):
        self.data_folder_path = data_folder_path
        self.data_files: Dict[int, str] = {}
        self._scan_data_files()
        
    def _scan_data_files(self):
        """Scans the data folder and maps z-values to file paths"""
        for file in os.listdir(self.data_folder_path):
            if file.endswith(('.tif', '.png', '.jpg')):
                # Extract number from filename
                parts = file.split('_')
                try:
                    if len(parts) > 1:
                        z_value = int(parts[-1].split('.')[0])  # Get last part before extension
                    else:
                        z_value = int(file.split('.')[0])  # Get whole name before extension
                    self.data_files[z_value] = os.path.join(self.data_folder_path, file)
                except ValueError:
                    print(f"Could not parse number from filename: {file}")
    
    def get_data(self, z_value: int) -> Optional[np.ndarray]:
        """Loads and returns the overlay for a specific z-value"""
        if z_value not in self.data_files:
            return None
        return cv2.imread(self.data_files[z_value], cv2.IMREAD_GRAYSCALE)
    
    @property
    def z_values(self) -> list[int]:
        """Returns sorted list of available z-values"""
        values = sorted(self.data_files.keys())
        if hasattr(self, 'z_range') and self.z_range:
            values = [z for z in values if z in self.z_range]
        return values
    
def process_single_z(z_value, z, zarr_size, ppm, ppm_mask, overlay_manager, surf_val, radius, dir):
    """Process a single z-value and write directly to zarr"""
    overlay = overlay_manager.get_data(z_value)
    if overlay is None:
        return
        
    # Get initial points
    rows, cols = np.where(overlay > 0)
    
    # Filter using ppm_mask
    if ppm_mask is not None:
        mask = ppm_mask[rows, cols] > 0
        rows = rows[mask]
        cols = cols[mask]
    
    # Get 3D points
    coords = np.stack([rows, cols], axis=-1).astype(np.float64)
    xyz_points = ppm.ijk_interpolator(coords)
    
    # Apply offset from normal
    if z_value != surf_val:
        normals = ppm.normal_interpolator(coords)
        xyz_points += dir * normals * (z_value - surf_val)
    
    # Convert to voxel coordinates and filter out-of-bounds points
    voxel_coords = xyz_points.astype(np.int32)
    mask = (
        (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < zarr_size[0]) &
        (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < zarr_size[1]) &
        (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < zarr_size[2])
    )
    
    valid_coords = voxel_coords[mask]
    valid_values = overlay[rows[mask], cols[mask]]
    
    # Write initial points
    z[valid_coords[:, 2], valid_coords[:, 1], valid_coords[:, 0]] = valid_values
    
    # Process and write neighbors
    neighbors = get_neighbors(radius)
    for dx, dy, dz in neighbors:
        neighbor_coords = valid_coords + np.array([dx, dy, dz])
        neighbor_mask = (
            (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < zarr_size[0]) &
            (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < zarr_size[1]) &
            (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < zarr_size[2])
        )
        
        valid_neighbors = neighbor_coords[neighbor_mask]
        if len(valid_neighbors) > 0:
            current_values = z[valid_neighbors[:, 2], valid_neighbors[:, 1], valid_neighbors[:, 0]]
            zero_mask = current_values == 0
            if np.any(zero_mask):
                z[valid_neighbors[zero_mask, 2], 
                  valid_neighbors[zero_mask, 1], 
                  valid_neighbors[zero_mask, 0]] = valid_values[neighbor_mask][zero_mask]

def parallel_depth_overlay_2d_to_3d_zarr(zarr_path, ppm_path, ppm_mask_path, overlay_folder_path, surf_val=32, 
                               zarr_size=(256,256,256), zarr_chunks=(128,128,128), z_range=None, 
                               radius=1, dir=1, num_workers=4):
    try:
        # Check if required files/folders exist
        if not os.path.exists(ppm_path):
            raise FileNotFoundError(f"PPM file not found: {ppm_path}")
        if not os.path.exists(ppm_mask_path):
            raise FileNotFoundError(f"PPM mask file not found: {ppm_mask_path}")
        if not os.path.exists(overlay_folder_path):
            raise FileNotFoundError(f"Overlay folder not found: {overlay_folder_path}")

        # Create/load zarr
        if not os.path.exists(zarr_path):
            zarr.create(shape=zarr_size, chunks=zarr_chunks, dtype=np.uint8, 
                       store=zarr_path, fill_value=0)
        
        z = zarr.open(zarr_path, mode='r+')
        if isinstance(z, zarr.hierarchy.Group):
            z = z[0]
        
        # Load PPM and mask
        ppm = Ppm.loadPpm(Path(ppm_path))
        ppm.loadData()
        ppm_mask = cv2.imread(ppm_mask_path, cv2.IMREAD_GRAYSCALE)
        overlay_manager = DataSliceManager(overlay_folder_path)
        overlay_manager.z_range = z_range
        
        print(f"Processing {len(overlay_manager.z_values)} overlay files using {num_workers} workers")
        
        # Create partial function with fixed arguments
        process_fn = partial(process_single_z, z=z, zarr_size=zarr_size, 
                           ppm=ppm, ppm_mask=ppm_mask, overlay_manager=overlay_manager,
                           surf_val=surf_val, radius=radius, dir=dir)
        
        # Process z-values in parallel
        with mp.Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(process_fn, overlay_manager.z_values),
                total=len(overlay_manager.z_values),
                desc="Processing z-values"
            ))
        
        print("Finished processing overlays")
        
    except Exception as e:
        error_msg = f"Error processing segment: {str(e)}"
        if __name__ == "__main__":
            raise Exception(error_msg)
        print(error_msg)
        return


def depth_overlay_2d_to_3d_pcd(ppm_path, ppm_mask_path, overlay_folder_path, surf_val=32, layers_folder_path=None, segment_id=None):
    ppm = Ppm.loadPpm(Path(ppm_path))
    ppm.loadData()
    ppm_mask = cv2.imread(ppm_mask_path, cv2.IMREAD_GRAYSCALE)
    overlay_manager = DataSliceManager(overlay_folder_path)
    if layers_folder_path is not None:
        layer_manager = DataSliceManager(layers_folder_path)
    print(f"Found {len(overlay_manager.z_values)} overlay files")
    print(overlay_manager.z_values)

    for z_value in overlay_manager.z_values:
        overlay = overlay_manager.get_data(z_value)
        layer = None
        if layers_folder_path is not None:
            layer = layer_manager.get_data(z_value)
        print(f"Overlay shape: {overlay.shape}")
        if overlay is not None:
            print(f"Creating point cloud for z-value: {z_value}")
            pcd = create_point_cloud(ppm, overlay, ppm_mask=ppm_mask, color=[0,1,0], z_diff=(z_value-surf_val), layer=layer)
            o3d.io.write_point_cloud(os.path.join('output', f"{segment_id}_pcd_overlay_{z_value}.ply"), pcd)

def parse_range(range_str):
    """Parse a range string like '30-35' into start and end values"""
    if not range_str:
        return None
    try:
        start, end = map(int, range_str.split('-'))
        return range(start, end + 1)  # +1 to include end value
    except ValueError:
        raise argparse.ArgumentTypeError('Range must be in format "start-end"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert 2D depth overlays to 3D representations')
    parser.add_argument('--base-path', type=str, 
                        default='/Users/jamesdarby/Desktop/test_segs/',
                        help='Base path for segment data')
    parser.add_argument('--segment-id', type=str,
                        default='20240301161650',
                        help='Segment ID')
    parser.add_argument('--surf-val', type=int, 
                        default=32,
                        help='Value of the layer/overlay slice that specifies the surface')
    parser.add_argument('--zarr-path', type=str,
                        default='/Users/jamesdarby/Documents/VesuviusScroll/GP/labels_2D_to_3D/s1_791um_label.zarr',
                        help='Path to zarr file location')
    parser.add_argument('--zarr-size', type=tuple,
                        default=(14376, 8096, 7888),
                        help='Size of the zarr array if needed to create')
    parser.add_argument('--zarr-chunks', type=tuple,
                        default=(128,128,128),
                        help='Chunk size of the zarr array if needed to create')
    parser.add_argument('--z-range', type=str,
                        help='Optional range of z-values to process (e.g., "30-35")')
    parser.add_argument('--overlay-subdir', type=str,
                        default='vr-hz-base',
                        help='Subdirectory name under overlays/ containing the overlay files')
    parser.add_argument('--radius', type=int,
                        default=0,
                        help='Radius of the additonal neighbourhood to include when projecting labels on normals')
    parser.add_argument('--dir', type=int,
                        default=1,
                        help='Direction to offset points along normals, -1 to flip direction')
    parser.add_argument('--num-workers', type=int,
                       default=4,
                       help='Number of worker processes for parallel processing')

    args = parser.parse_args()
    z_range = parse_range(args.z_range)

    ppm_mask_path = os.path.join(args.base_path, args.segment_id, f"{args.segment_id}_mask.png")
    # layers_folder_path = os.path.join(args.base_path, args.segment_id, "layers")
    overlay_folder_path = os.path.join(args.base_path, args.segment_id, "overlays", args.overlay_subdir)
    ppm_path = os.path.join(args.base_path, args.segment_id,f"{args.segment_id}.ppm")

    parallel_depth_overlay_2d_to_3d_zarr(
        args.zarr_path, 
        ppm_path, 
        ppm_mask_path, 
        overlay_folder_path, 
        args.surf_val, 
        args.zarr_size, 
        args.zarr_chunks, 
        z_range, 
        args.radius,
        args.dir,
        args.num_workers
    )
