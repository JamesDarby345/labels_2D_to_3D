import zarr
import dask.array as da

z_path = '/Users/jamesdarby/Documents/VesuviusScroll/GP/labels_2D_to_3D/s1_791um_label.zarr'
z = zarr.open(z_path, mode='r')

print(z.info)
print(z.dtype)
print(z.chunks) #128 chunks
print(z.shape)

z_da = da.from_zarr(z, chunks=(256,256,256))
print(z_da)

unique_vals = da.unique(z_da[8000:8256, :, :]).compute()
print("Unique values:", unique_vals)
