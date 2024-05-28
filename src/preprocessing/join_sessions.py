import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
from ScanImageTiffReader import ScanImageTiffReader
from scipy.signal import find_peaks
import tifffile as tiff


# Log continuous.
# ###############

path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\special_sessions\\AR099'

log1 = np.fromfile(os.path.join(path, 'Training', 'AR099_20230804_111449', 'log_continuous.bin'))
log2 = np.fromfile(os.path.join(path, 'Training', 'AR099_20230804_111449_UM', 'log_continuous.bin'))

# Cut last ttl up from first session due to stopping, fuse and save.
log = np.concatenate([log1[:-12000], log2])

save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\special_sessions\\AR099\\corrected\\Training\\AR099_20230804_111449\\log_continuous.bin'
with open(save_path, mode='wb') as fid:
    log.tofile(fid)


# Merging movies.
# ###############

# movies1 = ["D:\\AR\\AR129 24-03-01 13-28-58.avi", 'D:\\AR\\AR129 24-03-03 15-25-39.avi']
# movies2 = ["D:\\AR\\AR129 24-03-01 14-28-27.avi", 'D:\\AR\\AR129 24-03-03 16-10-25.avi']
# save_paths = ['D:\\AR\\AR129 24-03-01 13-28-58_merged.avi',
#               'D:\\AR\\AR129 24-03-03 15-25-39_merged.avi']

movie1 = "D:\\AR\\AR129 24-03-03 15-25-39.avi"
movie2 = "D:\\AR\\AR129 24-03-03 16-10-25.avi"
save_path = 'D:\\AR\\AR129 24-03-03 15-25-39_merged.avi'

videofiles = [movie1, movie2]

video_index = 0
cap = cv2.VideoCapture(videofiles[0])

fourcc = cv2.VideoWriter_fourcc(*'Y800')
out = cv2.VideoWriter(save_path, fourcc, 100.0, (640, 480))
count_frame = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    # print(count_frame, end='\r')
    # count_frame += 1
    if frame is None:
        print ("end of video " + str(video_index) + " .. next one now")
        video_index += 1
        if video_index >= len(videofiles):
            break
        cap = cv2.VideoCapture(videofiles[ video_index ])
        ret, frame = cap.read()
    # cv2.imshow('frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print ("end.")


# Solving mess with AR141.
# ########################

""" Solve the mess of pressing start instead of resume i.e. imaging continues while video filming and logging stops.
The strategy is to find a tiume stamp in the log file which will be the end of the session -
Take the end of the last logged trial - and delete the excess of imaging frame by
rewritting the tiff; similarly for the avi file.
 """


# read the log and determine final time stamp.

log_1 = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\training\\AR141_20240520_104531\\log_continuous.bin"
log_1 = np.fromfile(log_1)
ttl_events = find_peaks((log_1[2::6]>2)[1:].astype(np.float64) - (log_1[2::6]>2)[:-1].astype(np.float64), distance=100, prominence=1)[0]

cut = ttl_events[-1] + 6 * 5000  # 6 sec after the last trial start.

# plt.plot(log_1[3::6][23959394-5000*10:23959394+5000*60])
# plt.plot(log_1[0::6][23959394-5000*10:23959394+5000*60])
# plt.plot(log_1[1::6][23959394-5000*10:23959394+5000*60])
# plt.plot(log_1[2::6][23959394-5000*10:23959394+5000*60])

log_1_corrected = np.copy(log_1)
for i in range(6):
    log_1_corrected[i::6][cut:] = 0

ttl_events_cor = find_peaks((log_1_corrected[2::6]>2)[1:].astype(np.float64) - (log_1_corrected[2::6]>2)[:-1].astype(np.float64), distance=100, prominence=1)[0]
galvo_events_cor = find_peaks(log_1_corrected[1::6], distance=100, prominence=1)[0]
cam_events_cor = find_peaks(log_1_corrected[3::6], distance=10, prominence=1)[0]

# Checking.
plt.plot(log_1_corrected[3::6])
plt.scatter(cam_events_cor, log_1_corrected[3::6][cam_events_cor])
plt.plot(log_1_corrected[0::6])
plt.plot(log_1_corrected[1::6])
plt.scatter(galvo_events_cor, log_1_corrected[1::6][galvo_events_cor])
plt.plot(log_1_corrected[2::6])
plt.scatter(ttl_events_cor, log_1_corrected[2::6][ttl_events_cor])

n_frames_imaging = galvo_events_cor.size
n_frames_filming = cam_events_cor.size

save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\training\\AR141_20240520_104531\\log_continuous_1_cor.bin'
with open(save_path, mode='wb') as fid:
    log_1_corrected.tofile(fid)


# Count frames in CI movie.

session_1_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\AR141_20240520'
tiff_list = os.listdir(session_1_path)

nframes = 0 
for tiff in tiff_list:
    tiff = os.path.join(session_1_path, tiff)
    nframes += ScanImageTiffReader(tiff).shape()[0]

session_2_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\AR141_20240520_2\\AR141_20240520_00004.tif'
nframes_2 = ScanImageTiffReader(session_2_path).shape()[0]


# Count frames in avi.

movie_1 = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32.avi"
video_capture = cv2.VideoCapture(movie_1)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


# Results:
# n imaging frames in the first session 141495 the second one has 15315.
# n imaging frames to keep: 139504 (in the first session)
141495 - 139504
# n filming frames in avi 430336
# n frames to keep 425376

# Rewrite these files

movie1 = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32.avi"
save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32_cor.avi'

video_index = 0
cap = cv2.VideoCapture(movie1)

fourcc = cv2.VideoWriter_fourcc(*'Y800')
out = cv2.VideoWriter(save_path, fourcc, 100.0, (640, 480))
count_frame = 0
while(cap.isOpened()):
    if count_frame > 430336:
        break
    else:
        print(f'frame {count_frame}/430336', end='\r')
        ret, frame = cap.read()
        out.write(frame)
        count_frame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
out.release()
cv2.destroyAllWindows()
print ("Fin.")

# Count frames of corrected movie to check.
movie_1 = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32_cor.avi'
video_capture = cv2.VideoCapture(movie_1)
video_length_cor = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# cut ci tiff

read_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\imaging\\AR141_20240520_1\\AR141_20240520_00003.tif'
tiff = ScanImageTiffReader(tiff).data
tiff.shape
tiff = tiff[:-(141495-139504)]
save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\imaging\\AR141_20240520_1\\AR141_20240520_00003_cor.tif'
tiff.imsave(save_path, tiff)

