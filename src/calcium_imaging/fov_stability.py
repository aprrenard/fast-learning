import os

import numpy as np
from ScanImageTiffReader import ScanImageTiffReader


tiff_file = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\data\\AR127\\Recording\\Imaging\\AR127_20240221_133407\\AR127_20240221_150um_00001.tif"
beginning = ScanImageTiffReader(tiff_file).data[:1000];
end = ScanImageTiffReader(tiff_file).data[-1000:];



