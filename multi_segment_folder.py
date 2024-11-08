import os
import argparse
from tqdm import tqdm
from single_segment_folder import depth_overlay_2d_to_3d_zarr, parse_range
import multiprocessing as mp
from functools import partial

def process_single_segment(segment_id, base_path, zarr_path, zarr_size, zarr_chunks, z_range, surf_val, overlay_subdir, radius, dir):
    try:
        segment_path = os.path.join(base_path, segment_id)
        if not os.path.exists(segment_path):
            raise FileNotFoundError(f"Segment folder not found: {segment_path}")

        ppm_mask_path = os.path.join(base_path, segment_id, f"{segment_id}_mask.png")
        overlay_folder_path = os.path.join(base_path, segment_id, "overlays", overlay_subdir)
        ppm_path = os.path.join(base_path, segment_id,f"{segment_id}.ppm")

        depth_overlay_2d_to_3d_zarr(zarr_path, 
                                    ppm_path, 
                                    ppm_mask_path, 
                                    overlay_folder_path, 
                                    surf_val, 
                                    zarr_size, 
                                    zarr_chunks, 
                                    z_range, 
                                    radius, 
                                    dir)
    except Exception as e:
        print(f"Error processing segment {segment_id}: {str(e)}")

def process_multi_segment_folder(base_path, segment_ids, zarr_path, zarr_size, zarr_chunks, z_range, surf_val, overlay_subdir, radius, dir):
    if not segment_ids:
        segment_ids = [f.name for f in os.scandir(base_path) if f.is_dir()]

    # Create a partial function with all args except segment_id fixed
    process_fn = partial(process_single_segment, 
                        base_path=base_path,
                        zarr_path=zarr_path,
                        zarr_size=zarr_size,
                        zarr_chunks=zarr_chunks,
                        z_range=z_range,
                        surf_val=surf_val,
                        overlay_subdir=overlay_subdir,
                        radius=radius,
                        dir=dir)

    # Use all available CPUs
    num_processes = max(1, mp.cpu_count())
    
    with mp.Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_fn, segment_ids), 
                 total=len(segment_ids), 
                 desc="Processing segments"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple segment folders')
    parser.add_argument('--base-path', type=str,
                        default='/Users/jamesdarby/Desktop/test_segs/',
                        help='Base path containing multiple segment folders')
    parser.add_argument('--segment-ids', type=str, nargs='+',
                        default=['20230702185753', '20240301161650'],
                        help='Optional list of specific segment IDs to process. If not provided, processes all segments.')
    parser.add_argument('--surf-val', type=int, default=32,
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

    args = parser.parse_args()
    z_range = parse_range(args.z_range)

    process_multi_segment_folder(
        args.base_path, 
        args.segment_ids, 
        args.zarr_path, 
        args.zarr_size, 
        args.zarr_chunks, 
        z_range, 
        args.surf_val, 
        args.overlay_subdir, 
        args.radius, 
        args.dir
    )
