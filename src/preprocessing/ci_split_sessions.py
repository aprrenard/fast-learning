import os 

import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
import tifffile as tiff
import matplotlib.pyplot as plt
import yaml


mice_id = [
    # 'AR132',
    #         'AR133',
    #         'AR135',
    #         'AR137',
    #         'AR139',
    #         'AR127',
    #         'AR131',
    #         'AR143',
    #         'AR144',
    #         'AR163',
            'AR176',
            'AR177',
            'AR178',
            'AR179',
            'AR180']

for mouse_id in mice_id:

    imaging_folder = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\data\\{mouse_id}\\Recording\\Imaging'
    if not os.path.exists(imaging_folder):
        continue
    session_list = [session for session in os.listdir(imaging_folder)]
    session_list = [session for session in session_list if os.path.isdir(os.path.join(imaging_folder, session))]
    for session in session_list:
        print(session)

    # Read traces of concatenated sessions.
    suite2p_folder = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{mouse_id}\\suite2p\\plane0'
    F_raw = np.load(os.path.join(suite2p_folder, 'F_raw.npy'), allow_pickle=True)
    F_neu = np.load(os.path.join(suite2p_folder, 'F_neu.npy'), allow_pickle=True)
    F0_raw = np.load(os.path.join(suite2p_folder, 'F0_raw.npy'), allow_pickle=True)
    F0_cor = np.load(os.path.join(suite2p_folder, 'F0_cor.npy'), allow_pickle=True)
    dff = np.load(os.path.join(suite2p_folder, 'dff.npy'), allow_pickle=True)
    stat = np.load(os.path.join(suite2p_folder, 'stat.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_folder, 'iscell.npy'), allow_pickle=True)
    ops = np.load(os.path.join(suite2p_folder, 'ops.npy'), allow_pickle=True)

    counting_done = False
    frames_per_session = []
    
    # Check if already counted at conversion.
    for session in session_list:
        folder = rf'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data/{mouse_id}/{session}/suite2p/plane0/reg_tif'
        if os.path.exists(os.path.join(folder, 'movie_specs.yaml')):
            counting_done = True
            with open(os.path.join(folder, 'movie_specs.yaml'), 'r') as file:
                movie_specs = yaml.safe_load(file)
                frames_per_session.append(movie_specs['movie shape']['n_frames'])

    if not counting_done:
        if os.path.exists(os.path.join(suite2p_folder, 'frames_per_session.npy')):        
            frames_per_session = np.load(os.path.join(suite2p_folder, 'frames_per_session.npy'), allow_pickle=True)
        else:
            # Count frames per session.
            for session in session_list:
                path = os.path.join(imaging_folder, session)
                tif_paths = [os.path.join(path, itif) for itif in os.listdir(path) if os.path.splitext(itif)[1] in ['.tif', '.tiff']]
                tif_paths = sorted(tif_paths)
                nframes = 0
                for itif in tif_paths:
                    print(itif, end='\r')
                    shape = ScanImageTiffReader(itif).shape()
                    nframes += shape[0]
                frames_per_session.append(nframes)
            np.save(os.path.join(suite2p_folder, 'frames_per_session.npy'), frames_per_session, allow_pickle=True)
        frames_per_session = list(frames_per_session)

    # Split traces.
    starts = [0] + frames_per_session[:-1]
    starts = np.cumsum(starts)
    stops = np.cumsum(frames_per_session)
    start_stop = list(zip(starts, stops))
    print(start_stop, end='\n')

    for isession, session in enumerate(session_list):
        save_path = rf'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data/{mouse_id}/{session}/suite2p/plane0'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        a, b = start_stop[isession]
        np.save(os.path.join(save_path, 'F_raw.npy'), F_raw[:, a:b])
        np.save(os.path.join(save_path, 'F_neu.npy'), F_neu[:, a:b])
        np.save(os.path.join(save_path, 'F0_raw.npy'), F0_raw[:, a:b])
        np.save(os.path.join(save_path, 'F0_cor.npy'), F0_cor[:, a:b])
        np.save(os.path.join(save_path, 'dff.npy'), dff[:, a:b])
        np.save(os.path.join(save_path, 'stat.npy'), stat)
        np.save(os.path.join(save_path, 'iscell.npy'), iscell)
        np.save(os.path.join(save_path, 'ops.npy'), ops)
        


    # Move reg tifs to the sessions they belong to.
    # ---------------------------------------------

    if mouse_id == 'AR099':
        continue

    reg_tif_folder = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{mouse_id}\\suite2p\\plane0\\reg_tif'
    reg_tif_list = os.listdir(reg_tif_folder)
    reg_tif_list = [tif for tif in reg_tif_list if os.path.splitext(tif)[1] in ['.tif', '.tiff']]
    # Lexicographic ordering of tifs (bad padding from suite2p).
    f = lambda x: int(x[6:-10])
    reg_tif_list = sorted(reg_tif_list, key=f)
    reg_tif_list = [os.path.join(reg_tif_folder, tif) for tif in reg_tif_list]
    batch_size = ops.item()['batch_size']  # n frames per tif.
    isession = 0
    frames = stops

    for itif, tif in enumerate(reg_tif_list):
        session = session_list[isession]
        save_path = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{mouse_id}\\{session}\\suite2p\\plane0\\reg_tif'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        current_max_frame = frames[isession]
        frame_count = (itif + 1) * batch_size

        # Case were you don't need to split a tif.
        if (frame_count < current_max_frame) or (itif == len(reg_tif_list)-1):
            new_location = os.path.join(save_path, os.path.basename(tif))
            os.rename(tif, new_location)
        
        # Case where the last tif of a session would be the last image of a tif.
        # No need to split either but increment session.
        elif frame_count == current_max_frame:
            new_location = os.path.join(save_path, os.path.basename(tif))
            os.rename(tif, new_location)
            isession += 1

        # Case where you need to split a tif.
        elif frame_count > current_max_frame:
            # you hit a tiff that needs to be split
            cut_left = current_max_frame % batch_size
            tif_left = ScanImageTiffReader(tif).data(beg=0, end=cut_left)
            tif_right = ScanImageTiffReader(tif).data(beg=cut_left)
            
            # Set new location and names.
            left_path = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{mouse_id}\\{session}\\suite2p\\plane0\\reg_tif'
            left_tif_name = os.path.splitext(os.path.basename(tif))[0] + f'_{session}.tif'
            tif_left_path = os.path.join(left_path, left_tif_name)
            next_session = session_list[isession+1]
            right_path = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{mouse_id}\\{next_session}\\suite2p\\plane0\\reg_tif'
            if not os.path.exists(right_path):
                os.mkdir(right_path)
            right_tif_name = os.path.splitext(os.path.basename(tif))[0] + f'_{next_session}.tif'
            tif_right_path = os.path.join(right_path, right_tif_name)

            # Write split tif.
            tiff.imwrite(tif_left_path, tif_left)
            tiff.imwrite(tif_right_path, tif_right)

            # Count frames to check.
            frames_per_session[isession] += cut_left
            frames_per_session[isession+1] += 500 - cut_left
            isession += 1


