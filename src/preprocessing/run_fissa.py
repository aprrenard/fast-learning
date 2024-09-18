import os
import sys

import numpy as np
import scipy
import matplotlib.pyplot as plt
import fissa


def set_merged_roi_to_non_cell(stat, iscell):
    # Set merged cells to 0 in iscell.
    if 'inmerge' in stat[0].keys():
        for i, st in enumerate(stat):
            # 0: no merge; -1: input of a merge; index > 0: result of a merge.
            if st['inmerge'] not in [0, -1]:
                iscell[i][0] = 0.0

    return iscell


def compute_baseline(F, fs, window):

    # Parameters --------------------------------------------------------------
    nfilt = 30  # Number of taps to use in FIR filter
    fw_base = 1  # Cut-off frequency for lowpass filter, in Hz
    base_pctle = 5  # Percentile to take as baseline value

    # Main --------------------------------------------------------------------
    # Ensure array_like input is a numpy.ndarray
    F = np.asarray(F)

    # For short measurements, we reduce the number of taps
    nfilt = min(nfilt, max(3, int(F.shape[1] / 3)))

    if fs <= fw_base:
        # If our sampling frequency is less than our goal with the smoothing
        # (sampling at less than 1Hz) we don't need to apply the filter.
        filtered_f = F
    else:
        # The Nyquist rate of the signal is half the sampling frequency
        nyq_rate = fs / 2.0

        # Cut-off needs to be relative to the nyquist rate. For sampling
        # frequencies in the range from our target lowpass filter, to
        # twice our target (i.e. the 1Hz to 2Hz range) we instead filter
        # at the Nyquist rate, which is the highest possible frequency to
        # filter at.
        cutoff = min(1.0, fw_base / nyq_rate)

        # Make a set of weights to use with our taps.
        # We use an FIR filter with a Hamming window.
        b = scipy.signal.firwin(nfilt, cutoff=cutoff, window='hamming')

        # The default padlen for filtfilt is 3 * nfilt, but in case our
        # dataset is small, we need to make sure padlen is not too big
        padlen = min(3 * nfilt, F.shape[1] - 1)

        # Use filtfilt to filter with the FIR filter, both forwards and
        # backwards.
        filtered_f = scipy.signal.filtfilt(b, [1.0], F, axis=1,
                                           padlen=padlen)

    # Take a percentile of the filtered signal and windowed signal
    baseline = scipy.ndimage.percentile_filter(filtered_f, percentile=base_pctle, size=(1,(fs*2*window + 1)), mode='constant', cval=+np.inf)

    # Ensure filtering doesn't take us below the minimum value which actually
    # occurs in the data. This can occur when the amount of data is very low.
    baseline = np.maximum(baseline, np.nanmin(F, axis=1, keepdims=True))

    return baseline


def compute_dff(F_cor, F_raw, fs, window=60):
    '''
    F_cor: decontaminated traces, output of Fissa
    F_raw: raw traces extracted by Fissa (not suite2p)
    fs: sampling frequency
    window: running window size on each side of sample for percentile computation
    '''
    f0 = compute_baseline(F_raw, fs, window)
    dff = F_cor / f0

    return f0, dff


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

suite2p_folders = []
if longitudinal:
    for mouse_id in mice_ids:
        suite2p_folder = os.path.join(get_experimenter_analysis_folder(experimenter),
                                    mouse_id, 'suite2p', 'plane0')
        suite2p_folders.append(suite2p_folder)
else:
    for mouse_id in mice_ids:
        # List sessions with imaging.
        sessions = os.path.join(get_data_folder(), mouse_id, 'Recording', 'Imaging')
        sessions = [os.path.join(sessions, folder)
                        for folder in os.listdir(sessions)
                        if os.path.isdir(os.path.join(sessions, folder))]
        sessions = [os.path.split(folder)[1] for folder in sessions]
        for session_id in sessions:
            suite2p_folder = os.path.join(get_experimenter_analysis_folder(experimenter),
                                          mouse_id, session_id, 'suite2p', 'plane0')
            suite2p_folders.append(suite2p_folder)

for suite2p_folder in suite2p_folders:

    stat = np.load(os.path.join(suite2p_folder,'stat.npy'), allow_pickle = True)
    ops = np.load(os.path.join(suite2p_folder,'ops.npy'), allow_pickle = True).item()
    iscell = np.load(os.path.join(suite2p_folder,'iscell.npy'), allow_pickle = True)

    # Set merged roi's to non-cells.
    iscell = set_merged_roi_to_non_cell(stat, iscell)
    
    # The registered tifs in the reg-tif folder created by suite2p
    # are not lexicographically ordered. Reorder them and give list of
    # tifs as argument to Fissa.
    tif_path = os.path.join(suite2p_folder, 'reg_tif')
    reg_tif_list = os.listdir(tif_path)
    reg_tif_list = [tif for tif in reg_tif_list if os.path.splitext(tif)[1] in ['.tif', '.tiff']]
    f = lambda x: int(x[6:-10])
    reg_tif_list = sorted(reg_tif_list, key=f)
    reg_tif_list = [os.path.join(tif_path, tif) for tif in reg_tif_list]

    # Get image size
    Lx = ops['Lx']
    Ly = ops['Ly']

    # Get the cell ids
    ncells = len(stat)
    cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
    cell_ids = cell_ids[iscell[:,0]==1]  # only keep the ROIs that are actually cells.
    num_rois = len(cell_ids)

    # Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
    rois = [np.zeros((Ly, Lx), dtype=bool) for _ in range(num_rois)]

    for i, n in enumerate(cell_ids):
        # i is the position in cell_ids, and n is the actual cell number
        # Don't remove overlapping pixels of merges.
        if 'imerge' in stat[0].keys():
            if np.array(stat[n]['imerge']).any():
                ypix = stat[n]['ypix']
                xpix = stat[n]['xpix']
            else:
                ypix = stat[n]['ypix'][~stat[n]['overlap']]
                xpix = stat[n]['xpix'][~stat[n]['overlap']]
        else:
            ypix = stat[n]['ypix'][~stat[n]['overlap']]
            xpix = stat[n]['xpix'][~stat[n]['overlap']]

        if (np.sum(xpix) == 0 or np.sum(ypix) == 0):
            print(f'ROI {n} overlaps fully.')
        rois[i][ypix, xpix] = 1

    print(f'Running Fissa separation for {suite2p_folder}.')
    exp = fissa.Experiment(reg_tif_list, [rois], suite2p_folder)
    exp.separate(max_iter=20000)
    
    # TODO: use convergence result to remove cells that didn't converge.
    
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

    F0, dff = compute_dff(F_cor, F_raw, fs=ops['fs'], window=60)
    
    # Saving data.
    np.save(os.path.join(suite2p_folder, 'F_cor'), F_cor)
    np.save(os.path.join(suite2p_folder, 'F_raw'), F_raw)
    np.save(os.path.join(suite2p_folder, 'F0'), F0)
    np.save(os.path.join(suite2p_folder, 'dff'), dff)
    print(f'Data saved.')