import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from single_segment_folder import depth_overlay_2d_to_3d_zarr, depth_overlay_2d_to_3d_pcd, process_zarr_morphology

def process_multiple_segments(base_path: str, surf_val: int = 32, zarr_base_path: str = None, 
                            zarr_size: tuple = (14376, 8096, 7888), zarr_chunks: tuple = (128,128,128),
                            morph_labels: tuple = (1,2), mode: str = 'zarr'):
    """Process multiple segments from a base directory"""
    
    # Get all segment folders (assuming first level directories are segment IDs)
    segment_dirs = [d for d in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"Found {len(segment_dirs)} segments to process")
    
    # Process each segment
    for segment_id in tqdm(segment_dirs, desc="Processing segments"):
        print(f"\nProcessing segment: {segment_id}")
        
        # Construct paths
        segment_path = os.path.join(base_path, segment_id)
        ppm_mask_path = os.path.join(segment_path, f"{segment_id}_mask.png")
        layers_folder_path = os.path.join(segment_path, "layers")
        overlay_folder_path = os.path.join(segment_path, "overlays", "vr-hz-base")
        ppm_path = os.path.join(segment_path, f"{segment_id}.ppm")
        
        # Skip if required files don't exist
        if not all(os.path.exists(p) for p in [ppm_mask_path, overlay_folder_path, ppm_path]):
            print(f"Skipping segment {segment_id} - missing required files")
            continue
            
        try:
            if mode == 'zarr':
                if zarr_base_path is None:
                    print("zarr_base_path is required for zarr mode")
                    continue
                    
                # Create segment-specific zarr path
                segment_zarr_path = os.path.join(zarr_base_path, f"{segment_id}.zarr")
                
                # Process zarr
                depth_overlay_2d_to_3d_zarr(
                    segment_zarr_path, ppm_path, ppm_mask_path, overlay_folder_path,
                    surf_val, segment_id, zarr_size, zarr_chunks
                )
                
                # Apply morphological operations
                process_zarr_morphology(segment_zarr_path, labels=morph_labels)
                
            elif mode == 'pcd':
                # Create output directory if it doesn't exist
                output_dir = os.path.join(segment_path, 'output')
                os.makedirs(output_dir, exist_ok=True)
                
                # Process point cloud
                depth_overlay_2d_to_3d_pcd(
                    ppm_path, ppm_mask_path, overlay_folder_path,
                    surf_val, layers_folder_path, segment_id
                )
                
            else:
                print(f"Unknown mode: {mode}")
                continue
                
        except Exception as e:
            print(f"Error processing segment {segment_id}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple segment folders')
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path containing multiple segment folders')
    parser.add_argument('--surf-val', type=int, default=32,
                        help='Value of the layer/overlay slice that specifies the surface')
    parser.add_argument('--zarr-base-path', type=str,
                        help='Base path for zarr outputs')
    parser.add_argument('--zarr-size', type=tuple,
                        default=(14376, 8096, 7888),
                        help='Size of the zarr array if needed to create')
    parser.add_argument('--zarr-chunks', type=tuple,
                        default=(128,128,128),
                        help='Chunk size of the zarr array if needed to create')
    parser.add_argument('--morph-labels', type=tuple,
                        default=(1,2),
                        help='Label values to process with morphological operations')
    parser.add_argument('--mode', type=str, choices=['zarr', 'pcd'],
                        default='zarr',
                        help='Processing mode: zarr or point cloud')

    args = parser.parse_args()

    process_multiple_segments(
        args.base_path,
        args.surf_val,
        args.zarr_base_path,
        args.zarr_size,
        args.zarr_chunks,
        args.morph_labels,
        args.mode
    )
