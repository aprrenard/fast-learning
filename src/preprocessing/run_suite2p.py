import os

from suite2p import run_s2p, default_ops


EXPERIMENTER_MAP = {
    'AR': 'Anthony_Renard',
    'RD': 'Robin_Dard',
    'AB': 'Axel_Bisi',
    'MP': 'Mauro_Pulin',
    'PB': 'Pol_Bech',
    'MM': 'Meriam_Malekzadeh',
    'MS': 'Lana_Smith',
    'GF': 'Anthony_Renard',
    'MI': 'Anthony_Renard',
    }


def get_data_folder():
    data_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'data')

    return data_folder  


def get_experimenter_analysis_folder(initials):
    # Map initials to experimenter to get analysis folder path.
    experimenter = EXPERIMENTER_MAP[initials]
    analysis_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis',
                                   experimenter, 'data')
    return analysis_folder


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