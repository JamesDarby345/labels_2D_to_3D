import argparse
from dask.diagnostics import ProgressBar
import fastmorph
import dask.array as da
import numpy as np

def morph_close_chunks_zarr(zarr_path, chunk_size=512, radius=16, zarr_name="", morph_labels=[]):
    chunks = (chunk_size,chunk_size,chunk_size)
    data = da.from_zarr(zarr_path, chunks=chunks)
    if not zarr_name:
        zarr_name = zarr_path.split("/")[-1]
    print("saving to",zarr_name)
    parallel = 1

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
        arr = fastmorph.dilate(arr, parallel=parallel)
        arr = fastmorph.erode(arr, parallel=parallel)
        return arr[radius:-radius,radius:-radius,radius:-radius]

    
    closed_data = data.map_blocks(morph_close_chunk, dtype=data.dtype, meta=True)#.compute(scheduler='processes')

    with ProgressBar():
        closed_data.to_zarr(zarr_name, compute=False, overwrite=True).compute(scheduler='processes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform morphological closing on zarr arrays')
    parser.add_argument('--zarr-path', type=str, 
                      default="/Users/jamesdarby/Documents/VesuviusScroll/GP/labels_2D_to_3D/s1_791um_label.zarr",
                      help='Path to input zarr array')
    parser.add_argument('--chunk-size', type=int, default=512,
                      help='Size of dask chunks for morphological processing; recommened to be a multiple of the zarr chunk size')
    parser.add_argument('--radius', type=int, default=16,
                      help='Radius for morphological operation padding; keep at >16 due to fastmorph bug')
    parser.add_argument('--output-zarr', type=str, 
                      default="",
                      help='Output zarr array name')
    parser.add_argument('--morph-labels', nargs='*', type=int, default=[],
                      help='Label values to process (empty list processes all non-zero values)')

    args = parser.parse_args()

    morph_close_chunks_zarr(
        args.zarr_path, 
        args.chunk_size, 
        args.radius, 
        zarr_name=args.output_zarr, 
        morph_labels=args.morph_labels
    )
