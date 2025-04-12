
import os
import tifffile


def rewrite_tiff(input_file, output_file):
    # Read the TIFF file into memory
    print(f"Reading TIFF file {input_file}...")
    with tifffile.TiffFile(input_file) as tif:
        data = tif.asarray()  # Load the entire image into memory
    # Write the data to a new TIFF file
    with tifffile.TiffWriter(output_file, bigtiff=True) as tif_writer:
        tif_writer.write(data, photometric='minisblack')
    print(f"TIFF file has been successfully read and written to {output_file}.")
    

def rewrite_tiff_frame_by_frame(input_file, output_file, exclude_last_frame=False):
    # Read the TIFF file frame by frame
    print(f"Reading TIFF file {input_file} frame by frame...")
    with tifffile.TiffFile(input_file) as tif:
        frames = tif.pages
        num_frames = len(frames)
        if exclude_last_frame:
            num_frames -= 1  # Exclude the last frame if the option is enabled
        
        with tifffile.TiffWriter(output_file, bigtiff=True) as tif_writer:
            for i in range(num_frames):
                frame = frames[i].asarray()
                tif_writer.write(frame, photometric='minisblack')
    print(f"TIFF file has been successfully processed frame by frame and written to {output_file}.")



# # Path to the input and output TIFF files
# input_files = [
#     "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240519_151737/AR144_20240519_00001.tif",
    
#     "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240520_141104/AR144_20240520_00002.tif",
#     "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240520_141104/AR144_20240520_00003.tif",
#     "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240520_141104/AR144_20240520_00004.tif",
    
#     "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240521_142259/AR144_20240521_00001.tif",
    
#     "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240522_190834/AR144_20240522_00001.tif",
#     "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240522_190834/AR144_20240522_00002.tif",
# ]

# output_files = [
#     "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240519_151737/AR144_20240519_00001.tif",
    
#     "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240520_141104/AR144_20240520_00002.tif",
#     "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240520_141104/AR144_20240520_00003.tif",
#     "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240520_141104/AR144_20240520_00004.tif",
    
#     "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240521_142259/AR144_20240521_00001.tif",
    
#     "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240522_190834/AR144_20240522_00001.tif",
#     "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240522_190834/AR144_20240522_00002.tif",
# ]

# Path to the input and output TIFF files
input_files = [
    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241123_180709/AR163_20241123_210um_00001.tif",
    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241123_180709/AR163_20241123_210um_00002.tif",

    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241124_163658/AR163_20241124_00001.tif",

    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241125_153447/AR163_20241125_153447.tif",

    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241126_154140/AR163_20241126_00001.tif",
    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241126_154140/AR163_20241126_00002.tif",
    
    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241127_130218/AR163_20241127_00001.tif",
    # "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241127_130218/AR163_20241127_00002.tif",
    
    "/mnt/lsens-data/AR163/Recording/Imaging/AR163_20241128_145450/AR163_20241128_00001.tif",
]

output_files = [
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241123_180709/AR163_20241123_210um_00001.tif",
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241123_180709/AR163_20241123_210um_00002.tif",
    
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241124_163658/AR163_20241124_00001.tif",
    
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241125_153447/AR163_20241125_153447.tif",
    
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241126_154140/AR163_20241126_00001.tif",
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241126_154140/AR163_20241126_00002.tif",
    
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241127_130218/AR163_20241127_00001.tif",
    # "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241127_130218/AR163_20241127_00002.tif",
    
    "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR163/Recording/Imaging/AR163_20241128_145450/AR163_20241128_00001.tif",
]

for input_file, output_file in zip(input_files, output_files):
    # Create the output directory if it doesn't exist
    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)
    
    # Rewrite the TIFF file
    # rewrite_tiff(input_file, output_file)
    rewrite_tiff_frame_by_frame(input_file, output_file, exclude_last_frame=False)
