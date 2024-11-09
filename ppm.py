import pathlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import mmap

# PPM class for loading and interpolating PPM data
# Taken from Khartes implementation, thanks Chuck!
# Updated to provide chunked loading and interpolators for parallel processing
class Ppm():

    def __init__(self):
        self.mmap = None
        self.data = None
        self.ijks = None
        self.normals = None
        self.ijk_interpolator = None
        self.normal_interpolator = None
        self.data_header = None
        self.valid = False
        self.error = "no error message set"
        self.path = None
        self._header_size = 0

    def createErrorPpm(err):
        ppm = Ppm()
        ppm.error = err
        return ppm

    no_data = (0.,0.,0.)

    # lijk (layer ijk) is in layer's global coordinates
    def layerIjksToScrollIjks(self, lijks):
        print("litsi")
        if self.data is None:
            print("litsi no data")
            return lijks
        '''
        li,lj,lk = lijk
        if li < 0 or lj < 0:
            return Ppm.no_data
        if li >= self.width:
            return Ppm.no_data
        if lj >= self.height:
            return Ppm.no_data
        '''
        # sijk = np.zeros((lijk.shape), dtype=lijk.dtype)
        # sijks = self.ijk_interpolator(lijks[:,0:2])
        ijs = lijks[:,(2,0)]
        ks = lijks[:,1,np.newaxis]
        sijks = self.ijk_interpolator(ijs)
        norms = self.normal_interpolator(ijs)
        print(lijks.shape, sijks.shape, norms.shape, ks.shape)
        sijks += norms*(ks-32)
        return sijks

    def get_subsection(self, start_row, end_row, start_col, end_col):
        """Get a view of a subsection of the PPM data"""
        if self.data is None:
            self.loadData()
        
        subsection = np.ndarray((end_row - start_row, end_col - start_col, 6), 
                               dtype=np.float64,
                               buffer=self.mmap,
                               offset=self._header_size + (start_row * self.width + start_col) * 8 * 6,
                               strides=(self.width * 8 * 6, 8 * 6, 8))
        
        return subsection

    def loadData(self):
        if self.data is not None:
            return
        
        if not self.path.exists():
            return Ppm.createErrorPpm(f"ppm file {self.path} does not exist")

        try:
            fd = open(self.path, "rb")
            self.mmap = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Find header size
            header_end = self.mmap.find(b'<>\n')
            if header_end < 0:
                return Ppm.createErrorPpm("No header found")
            self._header_size = header_end + 3
            
            # Create memory mapped array
            data_size = self.height * self.width * 8 * 6
            self.data = np.ndarray((self.height, self.width, 6), 
                                 dtype=np.float64,
                                 buffer=self.mmap,
                                 offset=self._header_size)
            
            self.ijks = self.data[:,:,:3]
            self.normals = self.data[:,:,3:]
            
            # Create interpolators for the full dataset
            ii = np.arange(self.height)
            jj = np.arange(self.width)
            self.ijk_interpolator = RegularGridInterpolator(
                (ii, jj), self.ijks, fill_value=0., bounds_error=False)
            self.normal_interpolator = RegularGridInterpolator(
                (ii, jj), self.normals, fill_value=0., bounds_error=False)
            
        except Exception as e:
            return Ppm.createErrorPpm(f"Failed to mmap ppm file: {str(e)}")

    def __del__(self):
        if self.mmap is not None:
            self.mmap.close()

    # reads and loads the header of the ppm file
    def loadPpm(filename):
        fstr = str(filename)
        # print("reading ppm header for", filename)
        if not filename.exists():
            err="ppm file %s does not exist"%fstr
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            fd = filename.open("rb")
        except Exception as e:
            err="Failed to open ppm file %s: %s"%(fstr, e)
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            bstr = fd.read(200)
        except Exception as e:
            err="Failed to read ppm file %s: %s"%(fstr, e)
            print(err)
            return Ppm.createErrorPpm(err)

        index = bstr.find(b'<>\n')
        if index < 0:
            err="Ppm file %s does not have a header"%fstr
            print(err)
            return Ppm.createErrorPpm(err)

        hstr = bstr[:index+3].decode('utf-8')
        lines = hstr.split('\n')
        hdict = {}
        for line in lines:
            words = line.split()
            if len(words) != 2:
                continue
            name = words[0]
            value = words[1]
            if name[-1] != ':':
                continue
            name = name[:-1]
            hdict[name] = value
        for name in ["width", "height"]:
            if name not in hdict:
                err="Ppm file %s missing \"%s\" in header"%(fstr, name)
                print(err)
                return Ppm.createErrorPpm(err)

        try:
            width = int(hdict["width"])
        except Exception as e:
            err="Ppm file %s could not parse width value \"%s\" in header"%(fstr, hdict["width"])
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            height = int(hdict["height"])
        except Exception as e:
            err="Ppm file %s could not parse height value \"%s\" in header"%(fstr, hdict["height"])
            print(err)
            return Ppm.createErrorPpm(err)

        expected = {
                "dim": "6",
                "ordered": "true",
                "type": "double",
                "version": "1",
                }

        for name, value in expected.items():
            if name not in hdict:
                err = "Ppm file %s missing \"%s\" from header"%(fstr, name)
                print(err)
                return Ppm.createErrorPpm(err)
            if hdict[name] != expected[name]:
                err = "Ppm file %s expected value of \"%s\" for \"%s\" in header; got %s"%(fstr, expected[name], name, hdict[name])
                print(err)
                return Ppm.createErrorPpm(err)
    
        ppm = Ppm()
        ppm.valid = True
        ppm.height = height
        ppm.width = width
        ppm.path = filename
        ppm.name = filename.stem
        # print("created ppm %s width %d height %d"%(ppm.name, ppm.width, ppm.height))
        return ppm

    def loadChunk(self, start_row, end_row, start_col, end_col):
        """Load a chunk of PPM data and set up interpolators that work with global coordinates"""
        self.chunk_data = self.get_subsection(start_row, end_row, start_col, end_col)
        self.chunk_ijks = self.chunk_data[:,:,:3]
        self.chunk_normals = self.chunk_data[:,:,3:]
        
        # Create interpolators for the chunk that accept global coordinates
        chunk_rows = np.arange(start_row, end_row)
        chunk_cols = np.arange(start_col, end_col)
        
        self.chunk_ijk_interpolator = RegularGridInterpolator(
            (chunk_rows, chunk_cols), self.chunk_ijks, 
            fill_value=0., bounds_error=False)
        self.chunk_normal_interpolator = RegularGridInterpolator(
            (chunk_rows, chunk_cols), self.chunk_normals, 
            fill_value=0., bounds_error=False)

    def getLoadedChunk(self):
        """Return the currently loaded chunk data"""
        if not hasattr(self, 'chunk_data'):
            raise ValueError("No chunk loaded. Call loadChunk first.")
        return self.chunk_data