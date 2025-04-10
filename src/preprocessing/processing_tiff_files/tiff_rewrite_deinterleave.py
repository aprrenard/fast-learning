import os
import tifffile as tiff

def rewrite_tiff_deinterleave(input_path, output_path):
    # Open the input TIFF file
    with tiff.TiffFile(input_path) as tif:
        # Open the output TIFF file for writing
        with tiff.TiffWriter(output_path, bigtiff=True) as tif_writer:
            for i, page in enumerate(tif.pages):
                # if i>100:  # Testing.
                #     break
                print(f"Processing page {i}", end='\r')
                # Write only frames with an odd index (1-based odd frames)
                if i % 2 == 0:
                    tif_writer.write(page.asarray(), photometric='minisblack')

if __name__ == "__main__":
    input_file = "/mnt/lsens-data/AR144/Recording/Imaging/AR144_20240518_193553/AR144_20240518_00001.tif"
    output_folder = "/mnt/lsens-analysis/Anthony_Renard/need_fix/AR144/Recording/Imaging/AR144_20240518_193553"
    output_file = "AR144_20240518_193553.tif"  # Replace with your output file path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, output_file)
    rewrite_tiff_deinterleave(input_file, output_file)
