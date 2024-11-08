import argparse
from dask.diagnostics import ProgressBar
import fastmorph
import dask.array as da
import numpy as np
import zarr
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def load_zarr_compute_chunk(zarr, compute_chunk, compute_chunk_coords):
    z,y,x = compute_chunk_coords
    # Get the start coordinates for this compute chunk
    z_start = z * compute_chunk[0]
    y_start = y * compute_chunk[1] 
    x_start = x * compute_chunk[2]
    
    # Get the end coordinates
    z_end = min(z_start + compute_chunk[0], zarr.shape[0])
    y_end = min(y_start + compute_chunk[1], zarr.shape[1])
    x_end = min(x_start + compute_chunk[2], zarr.shape[2])

    # Load the full compute chunk region from zarr
    chunk_data = zarr[z_start:z_end, y_start:y_end, x_start:x_end]
    return chunk_data

def write_zarr_compute_chunk(zarr, data, compute_chunk, compute_chunk_coords):
    z,y,x = compute_chunk_coords
    # Get the start coordinates for this compute chunk
    z_start = z * compute_chunk[0]
    y_start = y * compute_chunk[1] 
    x_start = x * compute_chunk[2]
    
    # Get the end coordinates
    z_end = min(z_start + compute_chunk[0], zarr.shape[0])
    y_end = min(y_start + compute_chunk[1], zarr.shape[1])
    x_end = min(x_start + compute_chunk[2], zarr.shape[2])

    # Write the data to the compute chunk region in zarr
    zarr[z_start:z_end, y_start:y_end, x_start:x_end] = data

def process_chunk(coords, zarr_path, compute_chunks, radius, morph_labels, fast_morph_parallel=8):
    z, y, x = coords
    zarr_array = zarr.open(zarr_path, mode='r+')
    chunk_data = load_zarr_compute_chunk(zarr_array, compute_chunks, (z,y,x))
    
    # if morph labels is None or empty, process blocks with any nonzero values
    if morph_labels is None or not morph_labels:
        if not np.any(chunk_data):
            return
    elif len(morph_labels) == 1:
        if not (chunk_data == morph_labels[0]).any():
            return
    else: 
        if not np.isin(chunk_data, morph_labels).any():
            return
        
    arr = np.array(chunk_data)  
    arr = np.pad(arr, pad_width=radius, mode='constant', constant_values=0)
    arr = fastmorph.dilate(arr, parallel=fast_morph_parallel)
    arr = fastmorph.erode(arr, parallel=fast_morph_parallel)
    processed = arr[radius:-radius,radius:-radius,radius:-radius]
    
    write_zarr_compute_chunk(zarr_array, processed, compute_chunks, (z,y,x))

def dask_morph_close_chunks_zarr(zarr_path, chunk_size=512, radius=16, output_zarr="", morph_labels=[]):
    chunks = (chunk_size,chunk_size,chunk_size)
    data = da.from_zarr(zarr_path, chunks=chunks)
    if not output_zarr:
        output_zarr = "closed_" + zarr_path.split("/")[-1]
    print("saving to",output_zarr)
    fast_morph_parallel = 1

    def morph_close_chunk(block):
        # if morph labels is None or empty, process blocks with any nonzero values
        if morph_labels is None or not morph_labels:
            if not np.any(block):
                return block
        elif len(morph_labels) == 1:
            if not (block == morph_labels[0]).any():
                return block
        else: 
            if not np.isin(block, morph_labels).any():
                return block
            
        arr = np.array(block)  
        arr = np.pad(arr, pad_width=radius, mode='constant', constant_values=0)
        arr = fastmorph.dilate(arr, parallel=fast_morph_parallel)
        arr = fastmorph.erode(arr, parallel=fast_morph_parallel)
        return arr[radius:-radius,radius:-radius,radius:-radius]

    
    closed_data = data.map_blocks(morph_close_chunk, dtype=data.dtype, meta=True)#.compute(scheduler='processes')

    with ProgressBar():
        closed_data.to_zarr(output_zarr, compute=False, overwrite=True).compute(scheduler='processes')

def inplace_morph_close_chunks_zarr(zarr_path, chunk_size=512, radius=16, morph_labels=[]):
    compute_chunks = (chunk_size,chunk_size,chunk_size)
    zarr_array = zarr.open(zarr_path, mode='r+')
    zarr_chunk_size = zarr_array.chunks
    print(f"Zarr chunk size: {zarr_chunk_size}")
    zarr_shape = zarr_array.shape
    print(f"Zarr shape: {zarr_shape}")
    
    cz_max = zarr_shape[0]//compute_chunks[0] + 1
    cy_max = zarr_shape[1]//compute_chunks[1] + 1
    cx_max = zarr_shape[2]//compute_chunks[2] + 1
    print(f"Max chunk indices: {cz_max}, {cy_max}, {cx_max}")

    # Create list of all chunk coordinates
    chunk_coords = [(z,y,x) 
                   for z in range(cz_max)
                   for y in range(cy_max) 
                   for x in range(cx_max)]
    
    
    n_processes = mp.cpu_count()
    
    # Create partial function with fixed arguments
    process_chunk_partial = partial(
        process_chunk,
        zarr_path=zarr_path,
        compute_chunks=compute_chunks,
        radius=radius,
        morph_labels=morph_labels
    )

    with mp.Pool(n_processes) as pool:
        list(tqdm(pool.imap(process_chunk_partial, chunk_coords), total=len(chunk_coords)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform morphological closing on zarr arrays')
    parser.add_argument('--zarr-path', type=str, 
                      default="/Users/jamesdarby/Documents/VesuviusScroll/GP/labels_2D_to_3D/s1_791um_label.zarr",
                      help='Path to input zarr array')
    parser.add_argument('--chunk-size', type=int, default=512,
                      help='Size of dask chunks for morphological processing; recommened to be a multiple of the zarr chunk size')
    parser.add_argument('--radius', type=int, default=16,
                      help='Radius for morphological operation padding; keep at >16 due to fastmorph bug')
    parser.add_argument('--morph-labels', nargs='*', type=int, default=[],
                      help='Label values to process (empty list processes all non-zero values)')
    parser.add_argument('--output-zarr', type=str, 
                    default="",
                    help='Output zarr array name, if blank will inplace update the input zarr array')
    args = parser.parse_args()

    if args.output_zarr:
        #will create/overwrite the output zarr array
        dask_morph_close_chunks_zarr(args.zarr_path, 
                                     args.chunk_size, 
                                     args.radius, 
                                     morph_labels=args.morph_labels, 
                                     output_zarr=args.output_zarr)
    else:
        #will inplace update the input zarr array
        inplace_morph_close_chunks_zarr(args.zarr_path, 
                                       args.chunk_size, 
                                       args.radius, 
                                       morph_labels=args.morph_labels)
    
