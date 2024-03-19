import os
import sys

from suite2p import run_s2p, default_ops
sys.path.append('H:\\anthony\\repos\\NWB_analysis')
from nwb_utils import server_path


mice_ids = ['RD046']
experimenter = 'AR'
longitudinal = False

# set your options for running
ops = default_ops() # populates ops with the default options
ops['batch_size'] = 1000
ops['threshold_scaling'] = 1.0 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
ops['fs'] = 30
ops['tau'] = 1.25 # timescale of gcamp to use for deconvolution
ops['delete_bin'] = True
# Save reg tif for Fissa.
ops['reg_tif'] = True

dbs = []
if longitudinal:
    # Concatenate all sessions and run suite2p once per mouse.
    for mouse_id in mice_ids:
        tiff_folders = os.path.join(server_path.get_data_folder(), mouse_id, 'Recording', 'Imaging')
        tiff_folders = [os.path.join(tiff_folders, folder) for folder in os.listdir(tiff_folders)
                        if os.path.isdir(os.path.join(tiff_folders, folder))]
        # tiff_folders = ['D://AR//test_data']
        fast_disk = os.path.join('D:', 'suite2p', mouse_id)
        save_path = os.path.join(server_path.get_experimenter_analysis_folder(experimenter),
                                 mouse_id)
        if not os.path.exists(fast_disk):
            os.mkdir(fast_disk)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # db overwrites any ops (allows for experiment specific settings)
        db = {
            'h5py': [], # a single h5 file path
            'h5py_key': 'data',
            'data_path': tiff_folders, # a list of folders with tiffs
                                                    # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
            'fast_disk': fast_disk, # string which specifies where the binary file will be stored (should be an SSD)
            'save_path0': save_path,
            }
        dbs.append(db)
else:
    # Run suite2p for each session of each mouse.
    for mouse_id in mice_ids:
        tiff_folders = os.path.join(server_path.get_data_folder(), mouse_id, 'Recording', 'Imaging')
        tiff_folders = [os.path.join(tiff_folders, folder)
                        for folder in os.listdir(tiff_folders)
                        if os.path.isdir(os.path.join(tiff_folders, folder))]
        
        for folder in tiff_folders:
            session_id = os.path.split(folder)[1]
            fast_disk = os.path.join('D:', 'suite2p', mouse_id, session_id)
            save_path = os.path.join(server_path.get_experimenter_analysis_folder(experimenter),
                                     mouse_id, session_id)
            if not os.path.exists(fast_disk):
                os.makedirs(fast_disk)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # db overwrites any ops (allows for experiment specific settings)
            db = {
                'h5py': [], # a single h5 file path
                'h5py_key': 'data',
                'data_path': [folder], # a list of folders with tiffs
                                                        # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                'fast_disk': fast_disk, # string which specifies where the binary file will be stored (should be an SSD)
                'save_path0': save_path,
                }
            dbs.append(db)
    
for dbi in dbs:
    opsEnd = run_s2p(ops=ops, db=dbi)
