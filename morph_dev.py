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

def sum_zarr(zarr_path, kernel_size=3, labels=None, chunk_size=512, batch_size=3, scheduler='processes'):

    # zarr_array = zarr.open_array(zarr_path)# Load zarr as dask array
    # z_array = zarr.open(zarr_path, mode='r')
    chunks = (chunk_size,chunk_size,chunk_size)
    data = da.from_zarr(zarr_path, chunks=chunks)
    print(f"Data shape: {data.shape}")
    print(f"Data chunks: {data.chunks}")
    print(f"Data dtype: {data.dtype}")
    print(f"Number of chunks: {data.npartitions}")
    # print(f"Memory usage: {data.nbytes / 1e9:.2f} GB")    

    # Define function to sum each block
    def chunk_sum(block):
        # Return array with same number of dimensions but size 1 in each dimension
        return np.array([block.sum()]).reshape((1,) * block.ndim)
    
    # Map the sum operation across all chunks and compute
    with ProgressBar():
        chunk_sums = data.map_blocks(chunk_sum, dtype=data.dtype).compute()
        print(chunk_sums.shape)
        print(chunk_sums)
    # Convert to list
    return chunk_sums.ravel().tolist()

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Morphological operations on zarr files')

    parser.add_argument('--zarr-path', type=str,
                            default='/Users/jamesdarby/Documents/VesuviusScroll/GP/labels_2D_to_3D/s1_791um_label.zarr',
                            help='Path to zarr file location')

    args = parser.parse_args()

    process_large_zarr_morphology(args.zarr_path, kernel_size=3, labels=[255])

