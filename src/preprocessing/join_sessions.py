import os

import numpy as np
import matplotlib.pyplot as plt
import cv2


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
