import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import fissa

sys.path.append('H://anthony//repos//fast-learning//src')
import server_path


mice_ids = ['ARXXX']

for mouse_id in mice_ids:
    output_folder = os.path.join(server_path.get_experimenter_analysis_folder('AR'),
                                 mouse_id, 'suite2p', 'plane0')
    # output_folder = os.path.join('D:\AR\calcium_imaging_processing\ARXXX', 'fissa')

    if not os.path.join(output_folder):
        os.mkdir(output_folder)

    suite2p_folder = os.path.join(server_path.get_experimenter_analysis_folder(experimenter),
                                  mouse_id, 'suite2p', 'plane0')
    stat = np.load(os.path.join(suite2p_folder,'stat.npy'), allow_pickle = True)
    ops = np.load(os.path.join(suite2p_folder,'ops.npy'), allow_pickle = True).item()
    iscell = np.load(os.path.join(suite2p_folder,'iscell.npy'), allow_pickle = True)[:,0]
    images = os.path.join(suite2p_folder, 'reg_tif')

    # Make sure to correct is_cell for merges.
    if 'inmerge' in stat[0].keys():  # If no merge in session, 'inmerge' not in dict.
        for i in range(len(stat)):
            if stat[i]['inmerge'] not in [0.0, -1.0]:
                iscell[i] = 0.0
        print('Merged ROIs corrected after merging..')

    # Get image size
    Lx = ops['Lx']
    Ly = ops['Ly']

    # Get the cell ids
    ncells = len(stat)
    cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
    cell_ids = cell_ids[iscell==1]  # only keep the ROIs that are actually cells.
    num_rois = len(cell_ids)

    # Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
    rois = [np.zeros((Ly, Lx), dtype=bool) for _ in range(num_rois)]

    for i, n in enumerate(cell_ids):
        # i is the position in cell_ids, and n is the actual cell number
        # Don't remove overlapping pixels of merges.
        if ('imerge' in stat[0].keys()) & (np.array(stat[n]['imerge']).any()):
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
        else:
            ypix = stat[n]['ypix'][~stat[n]['overlap']]
            xpix = stat[n]['xpix'][~stat[n]['overlap']]
        if (np.sum(xpix) == 0 or np.sum(ypix) == 0):
            print(f'ROI {n} overlaps fully.')
        rois[i][ypix, xpix] = 1

    print(f'Running Fissa separation for {mouse_id}.')
    exp = fissa.Experiment(images, [rois], output_folder)
    exp.separate()

    # Extract and reshape corrected traces to (ncells, nt).
    ncells, ntifs = exp.result.shape
    nt = exp.result[0,0][0].shape[0]
    F_cor = []
    for icell in range(ncells):
        tmp = []
        for itif in range(ntifs):
            tmp.append(exp.result[icell,itif][0])
        F_cor.append(np.concatenate(tmp))
    F_cor = np.vstack(F_cor)

    # Same for raw traces.
    ncells, ntifs = exp.result.shape
    nt = exp.raw[0,0][0].shape[0]
    F_raw = []
    for icell in range(ncells):
        tmp = []
        for itif in range(ntifs):
            tmp.append(exp.raw[icell,itif][0])
        F_raw.append(np.concatenate(tmp))
    F_raw = np.vstack(F_raw)

    # Saving data.
    np.save(os.path.join(output_folder, 'F_cor'), F_cor)
    np.save(os.path.join(output_folder, 'F_raw'), F_raw)

    # TODO: compute F0 and dff.