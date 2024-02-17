import os
import sys

import numpy as np

from suite2p import run_s2p, default_ops
sys.path.append('H://anthony//repos//fast-learning//src')
import server_path

mice_ids = ['AR103','AR104','AR106','AR107','AR071','AR099']
exclude = ['AR071_20230529_130617']
experimenter = 'AR'

dbs = []
for mouse_id in mice_ids:
  # set your options for running
  ops = default_ops() # populates ops with the default options

  ops['batch_size'] = 800
  ops['threshold_scaling'] = 1.0 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
  ops['fs'] = 30
  ops['tau'] = 1.25 # timescale of gcamp to use for deconvolution
  ops['delete_bin'] = True
  # Save reg tif for Fissa.
  ops['reg_tif'] = True
  ops['save_path0'] = os.path.join(server_path.get_experimenter_analysis_folder(experimenter),
                                   mouse_id)

  tiff_folders = os.path.join(server_path.get_data_folder(), mouse_id, 'Recording', 'Imaging')
  tiff_folders = [os.path.join(tiff_folders, folder) for folder in os.listdir(tiff_folders)
                  if os.path.isdir(os.path.join(tiff_folders, folder)) and folder not in exclude]
  # tiff_folders = ['D://AR//test_data']
  fast_disk = os.path.join('D:', 'suite2p', mouse_id)
  if not os.path.exists(fast_disk):
     os.mkdir(fast_disk)

  # db overwrites any ops (allows for experiment specific settings)
  db = {
        'h5py': [], # a single h5 file path
        'h5py_key': 'data',
        'data_path': tiff_folders, # a list of folders with tiffs 
                                              # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)          
        'fast_disk': fast_disk # string which specifies where the binary file will be stored (should be an SSD)
      }

  dbs.append(db)

for dbi in dbs:
    opsEnd = run_s2p(ops=ops, db=dbi)