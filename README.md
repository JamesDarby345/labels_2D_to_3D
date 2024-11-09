# 2D -> 3D

## Purpose of this repo
The purpose of this repo is to allow for the mapping of 2d labels with depth from the segments, 
such as fibre labels, surface labels, ink labels back to 3d using pre-existing .ppm files. 
The existing 3D ink detection, fibre and surface models simply projecting a single flat image
a certain depth instead of having slice by slice accuracy. This repo should make using depth 
annotated labels from the flattened segments faster and easier to train 3d models with more accurate labels.

## Installation
This repo assumes you have conda installed for the installation.

Run these commands to setup the repo:
```
git clone https://github.com/JamesDarby345/labels_2D_to_3D.git
cd labels_2d_to_3d
conda env create -f environment.yml
conda activate 2d3d
```

## Data format & Usage
The main file, `ppm_2d3d.py` is expecting a folder structure similar
to the segments on the Vesuvius download server. Ex:

```
20240301161650
  - 20240301161650.ppm
  - 20240301161650_mask.png
  layers
    - 00.tif
      ...
    - 32.tif
    - 33.tif
      ...
    - 64.tif
  overlays
    overlay-subdir
      - xxxxxxx_32.png
      - xxxxxxx_33.tif
      - xxxxxxx_34.jpg
```

The `ppm_2d3d.py` file uses the .ppm file to map the 2d coordinates to 3d space, and the overlay number to project the overlay label along the .ppm normal to the correct position. The output is saved to a zarr. 
It takes in a `--surf-val` parameter, the number of the .tif defining the surface of the .ppm for this calculation
(the default is 32, which is correct for the 7.91um segment, 64 for the 3.24um segments).

**It also takes:**<br>
`--base-path` which should be the path to the folder containing the segment folders<br>
`--segment-ids` to specify which segment id's it should run on, you can use "all" to run on all the segments in the base path<br>
`--zarr-path` to specify the path to the zarr to update or create with the new ppm mapped labels<br>
If a zarr doesnt exist at the specified path, it uses `--zarr-size` and `--zarr-chunks` to initialise a new empty zarr<br>
`--z-range` optionally limits the range of overlay values to use<br>
`--overlay-subdir` specifies which subdirectories in the overlays folder to use as the overlays to map, useful for different label sets (ink, fibres surface etc)<br>
`--chunk-size` defines how large each parralelised chunk should be, the memory requirements of the code will depend on how
large this value is and how many processes (cpu corse by default) are running. The default of 2048 works well
`--num_workers` defines how many processes to run, by default it is 0, which will use all available cpu cores. If you are running out of RAM, reduces this to use fewer processes, or reduce the chunk_size.

The other arguments are documented in the code and are less important

An example command running the file with most arguments would be like this, though you can also change the default values in the 
.py file if you prefer

```
python ppm_2d3d.py --base-path path/to/segments --segment-ids 20240301161650, 20230702185753 --zarr-path path/to/overlay.zarr
--z-range "30-35" --overlay-subdir surface-labels --zarr-size (14376, 8096, 7888) --zarr-chunks (128,128,128)
```

Both these files run utilising all the CPU cores, and use progress bars to display their progress. I may release a v2 using CUDA
to accelerate this operation even further perhaps.

The resulting zarr uses uint8 and is fairly sparse, so should be around 1~100GB depending on how many segments you map. The
code can map more segments/overlays to an already existing overlay zarr. Operation on one of the smaller segments with 3 overlays 
took ~ 90s on a macbook pro M2 Max. the values in the zarr will be the same as the value of the label in the overlay image.

The bigger the segment, the more segments and the more overlay z values used the longer it will take.

## Results
To visualise the resulting zarr array overtop of the original scan data, I recommend using [Vesuvius-GUI](https://github.com/jrudolph/vesuvius-gui) by jrudolph.
It dynamically downloads data so you dont need the original zarr and the resulting label zarr can be provided as an overlay zarr.
The results typically have small holes in the label as pictured below. I beleive this is a part of the .ppm mapping, and may not be ideal.

<img width="619" alt="Screenshot 2024-11-08 at 2 34 30 PM" src="https://github.com/user-attachments/assets/d7bf4279-1417-45a5-92d8-b68032875e8d">
<br>

To fix this and create labels without random small pixel holes, I developed and have included a `morph_close_zarr.py` script that 
will morphologically close the zarr labels so they look like this:

<img width="619" alt="Screenshot 2024-11-08 at 2 34 50 PM" src="https://github.com/user-attachments/assets/46924d3a-fcf8-450e-844b-24d9600dce1b">
<br> Notice the lack of single missing voxels

## morph_close_zarr.py usage
Note that this uses the fastmorph library for fast CPU morphologcial operations, it works on macOS but is untested on other OS's.
I may update this in the future and try implementing CUDA accelerated options.

The `morph_close_zarr.py` file takes in these command line arguments<br>
`--zarr-path` to specify the path on the input zarr to apply morphological closing to<br>
`--chunk-size` to specify the chunk size to apply morphology to in a batch, ideally a multiple of your zarr chunk size<br>
`--morph-labels` to specify which label values to apply morphological closing to, leave blank for all non-zero values. [255], [1,2] etc<br>
`--output-zarr` to specify the path to the output zarr. Leave blank to update the input zarr.<br>

This code should be using all of your CPU cores after a few seconds. It skips chunks where not labels exist. 
Depending on the amount of labels to morphologically close and hardware, it takes a few minutes or more.






