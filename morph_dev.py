import os
import numpy as np
import argparse
from pathlib import Path
import open3d as o3d
from ppm import Ppm
from typing import Optional, Dict
import zarr
import dask
import dask.array as da
from tqdm import tqdm
from dask_image.ndmorph import binary_closing
from dask.diagnostics import ProgressBar

def process_large_zarr_morphology(zarr_path, kernel_size=3, labels=None, chunk_size=256, batch_size=3, scheduler='processes'):

    # Configure dask to use processes
    with dask.config.set(scheduler=scheduler):
        # Load data as dask array with chunks
        data = da.from_zarr(zarr_path, chunks=(chunk_size, chunk_size, chunk_size))
        kernel = np.ones((kernel_size, kernel_size, kernel_size), np.uint8)
        
        # Get labels if not provided
        if labels is None:
            unique_labels = np.unique(data)
            labels = unique_labels[unique_labels > 0]
        elif isinstance(labels, (int, np.integer)):
            labels = (labels,)

        # Get array dimensions
        shape = data.shape
        n_chunks = [shape[i] // chunk_size + bool(shape[i] % chunk_size) for i in range(3)]
        print((n_chunks[0]), (n_chunks[1]), (n_chunks[2]))
        # Process chunks
        for z in range(n_chunks[0]):
            for y in range(n_chunks[1]):
                for x in range(n_chunks[2]):
                    # Get chunk coordinates
                    z_start = z * chunk_size
                    y_start = y * chunk_size 
                    x_start = x * chunk_size
                    z_end = min(z_start + chunk_size, shape[0])
                    y_end = min(y_start + chunk_size, shape[1])
                    x_end = min(x_start + chunk_size, shape[2])

                    print(f"Processing chunk ({z},{y},{x})")
                    chunk = data[z_start:z_end, y_start:y_end, x_start:x_end]

                    # Process labels in batches for this chunk
                    for i in range(0, len(labels), batch_size):
                        batch_labels = labels[i:i+batch_size]
                        
                        for label in batch_labels:
                            print(f"Processing label {label}")
                            mask = chunk == label
                            closed = binary_closing(mask, kernel)
                            chunk = da.where(closed, label, chunk)
                        
                        # Compute and write back chunk results
                        with ProgressBar():
                            print("Computing chunk")
                            chunk_result = chunk.compute()
                            data[z_start:z_end, y_start:y_end, x_start:x_end] = chunk_result

        # Write final results
        with ProgressBar():
            print("Writing to zarr") 
            data.to_zarr(zarr_path, overwrite=True)

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Morphological operations on zarr files')

    parser.add_argument('--zarr-path', type=str,
                            default='/Users/jamesdarby/Documents/VesuviusScroll/GP/labels_2D_to_3D/s1_791um_label.zarr',
                            help='Path to zarr file location')

    args = parser.parse_args()

    process_large_zarr_morphology(args.zarr_path, kernel_size=3, labels=[255])