import os
import numpy as np
import argparse
from pathlib import Path
from ppm import Ppm
from typing import Optional, Dict
import zarr
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from PIL import Image

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
        with Image.open(self.data_files[z_value]) as img:
            return np.array(img.convert('L'))
        
    def get_data_chunk(self, z_value: int, start_x: int, end_x: int, start_y: int, end_y: int) -> Optional[np.ndarray]:
        """Loads and returns a chunk of the image specified by the coordinates."""
        if z_value not in self.data_files:
            return None
        # Should only load the crop region into memory
        with Image.open(self.data_files[z_value]) as img:
            # Define the bounding box for the region of interest
            box = (start_x, start_y, end_x, end_y)
            # Crop the image to the defined box
            region = img.crop(box)
            # Convert to grayscale and numpy array
            region_array = np.array(region.convert('L'))
        return region_array
    
    @property
    def z_values(self) -> list[int]:
        """Returns sorted list of available z-values"""
        values = sorted(self.data_files.keys())
        if hasattr(self, 'z_range') and self.z_range:
            values = [z for z in values if z in self.z_range]
        return values
    
def generate_chunks(width: int, height: int, chunk_size: int):
    """Generate (start_x, end_x, start_y, end_y) coordinates for chunks"""
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunks.append((
                x, 
                min(x + chunk_size, width),
                y, 
                min(y + chunk_size, height)
            ))
    return chunks

def process_chunk(chunk_coords, z_values, z, zarr_size, ppm_path, ppm_mask, overlay_manager, surf_val, dir):
    """Process a single spatial chunk across all z-values"""
    # Load just the PPM header
    ppm = Ppm.loadPpm(Path(ppm_path))
    start_x, end_x, start_y, end_y = chunk_coords
    
    # Load only the chunk we need
    ppm.loadChunk(start_y, end_y, start_x, end_x)
    
    # Get PPM mask subsection for this chunk
    ppm_mask_chunk = ppm_mask[start_y:end_y, start_x:end_x] if ppm_mask is not None else None
    
    for z_value in z_values:
        overlay_chunk = overlay_manager.get_data_chunk(z_value, start_x, end_x, start_y, end_y)
        if overlay_chunk is None:
            continue
            
        # Get initial points within chunk
        rows, cols = np.where(overlay_chunk > 0)
        if len(rows) == 0:
            continue
            
        # Adjust coordinates to global space
        rows += start_y
        cols += start_x
        
        # Filter using ppm_mask
        if ppm_mask_chunk is not None:
            mask = ppm_mask_chunk[rows - start_y, cols - start_x] > 0
            rows = rows[mask]
            cols = cols[mask]
        
        # Use chunk interpolators with global coordinates
        coords = np.stack([rows, cols], axis=-1).astype(np.float64)
        xyz_points = ppm.chunk_ijk_interpolator(coords)
        
        if z_value != surf_val:
            normals = ppm.chunk_normal_interpolator(coords)
            xyz_points += dir * normals * (z_value - surf_val)
        
        voxel_coords = xyz_points.astype(np.int32)
        mask = (
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < zarr_size[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < zarr_size[1]) &
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < zarr_size[2])
        )
        
        valid_coords = voxel_coords[mask]
        valid_values = overlay_chunk[rows[mask] - start_y, cols[mask] - start_x]
        
        # Write to zarr
        z[valid_coords[:, 2], valid_coords[:, 1], valid_coords[:, 0]] = valid_values

def parallel_depth_overlay_2d_to_3d_zarr(zarr_path, ppm_path, ppm_mask_path, overlay_folder_path, surf_val=32, 
                               zarr_size=(256,256,256), zarr_chunks=(128,128,128), z_range=None, 
                               dir=1, num_workers=4, chunk_size=1024):
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
        ppm_mask = np.array(Image.open(ppm_mask_path).convert('L'))
        overlay_manager = DataSliceManager(overlay_folder_path)
        overlay_manager.z_range = z_range
        
        # Generate chunks based on PPM dimensions
        chunks = generate_chunks(ppm.width, ppm.height, chunk_size)
        
        print(f"Processing {len(chunks)} chunks using {num_workers} workers")
        
        # Create partial function with fixed arguments
        process_fn = partial(process_chunk, 
                            z_values=overlay_manager.z_values,
                            z=z, zarr_size=zarr_size, 
                            ppm_path=ppm_path, ppm_mask=ppm_mask, 
                            overlay_manager=overlay_manager,
                            surf_val=surf_val, dir=dir)
        
        # Process chunks in parallel
        with mp.Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(process_fn, chunks),
                total=len(chunks),
                desc="Processing chunks"
            ))
        
        print("Finished processing overlays")
        
    except Exception as e:
        error_msg = f"Error processing segment: {str(e)}"
        if __name__ == "__main__":
            raise Exception(error_msg)
        print(error_msg)
        return

def parse_range(range_str):
    """Parse a range string like '30-35' into start and end values"""
    if not range_str:
        return None
    try:
        start, end = map(int, range_str.split('-'))
        return range(start, end + 1)  # +1 to include end value
    except ValueError:
        raise argparse.ArgumentTypeError('Range must be in format "start-end"')

def get_all_segment_ids(base_path: str) -> list[str]:
    """Get all folder names in base path as segment IDs"""
    return [d for d in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, d))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert 2D depth overlays to 3D representations')
    parser.add_argument('--base-path', type=str, 
                        default='/Users/jamesdarby/Desktop/test_segs',
                        help='Base path for segment data')
    parser.add_argument('--segment-ids', type=str,
                        default='20240301161650, 20230702185753',
                        help='Segment ID(s). Use comma-separated values for multiple IDs, or "all" to process all segments')
    parser.add_argument('--surf-val', type=int, 
                        default=32,
                        help='Value of the layer/overlay slice that specifies the surface')
    parser.add_argument('--zarr-path', type=str,
                        default='/Users/jamesdarby/Documents/VesuviusScroll/GP/labels_2D_to_3D/s1_791um_label.zarr',
                        help='Path to zarr file location')
    parser.add_argument('--zarr-size', type=tuple,
                        default=(14376, 8096, 7888),
                        help='Size of the zarr array if needed to create')
    '''
    s1_volume_shape = (14376, 8096, 7888)    # scroll1_791um (z,y,x) 
    s2_volume_shape = (14428, 10112, 11984)  # scroll2_791um (z,y,x)
    s3_volume_shape = (9778, 3400, 3550)     # scroll3_791um (z,y,x)
    s4_volume_shape = (11174, 3440, 3340)    # scroll4_791um (z,y,x)
    '''
    parser.add_argument('--zarr-chunks', type=tuple,
                        default=(128,128,128),
                        help='Chunk size of the zarr array if needed to create')
    parser.add_argument('--z-range', type=str,
                        help='Optional range of z-values to process (e.g., "30-35")')
    parser.add_argument('--overlay-subdir', type=str,
                        default='surf-mask',
                        help='Subdirectory name under overlays/ containing the overlay files')
    parser.add_argument('--dir', type=int,
                        default=1,
                        help='Direction to offset points along normals, -1 to flip direction')
    parser.add_argument('--num-workers', type=int,
                       default=0,
                       help='Number of worker processes for parallel processing. <1 for num cpu cores')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Size of 2D PPM chunks for parallel processing')

    args = parser.parse_args()
    z_range = parse_range(args.z_range)
    Image.MAX_IMAGE_PIXELS = None

    # Handle multiple segment IDs
    if args.segment_ids.lower() == 'all':
        segment_ids = get_all_segment_ids(args.base_path)
    else:
        segment_ids = [s.strip() for s in args.segment_ids.split(',')]

    if args.num_workers < 1:
        args.num_workers = mp.cpu_count()

    # Process each segment sequentially
    for segment_id in segment_ids:
        print(f"\nProcessing segment: {segment_id}")
        ppm_mask_path = os.path.join(args.base_path, segment_id, f"{segment_id}_mask.png")
        overlay_folder_path = os.path.join(args.base_path, segment_id, "overlays", args.overlay_subdir)
        ppm_path = os.path.join(args.base_path, segment_id, f"{segment_id}.ppm")
        
        try:
            parallel_depth_overlay_2d_to_3d_zarr(
                args.zarr_path, 
                ppm_path, 
                ppm_mask_path, 
                overlay_folder_path, 
                args.surf_val, 
                args.zarr_size, 
                args.zarr_chunks, 
                z_range, 
                args.dir,
                args.num_workers,
                args.chunk_size
            )
        except Exception as e:
            print(f"Error processing segment {segment_id}: {str(e)}")
            continue
