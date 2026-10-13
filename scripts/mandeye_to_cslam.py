# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:12:08 2025

@author: Pawel
"""

import os
import shutil
import argparse

def sync_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cam0_folder = os.path.join(output_folder, "cam0")
    cam1_folder = os.path.join(output_folder, "cam1")
    os.makedirs(cam0_folder, exist_ok=True)
    os.makedirs(cam1_folder, exist_ok=True)
    
    folders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.startswith("photos_") and os.path.isdir(os.path.join(input_folder, f))]
    pairs_count = 0
    for folder in folders:
        timestamps = set([f.replace('.jpg',"").split("_")[-1] for f in os.listdir(folder)])
        for timestamp in timestamps:
            im0 = os.path.join(folder, f"camera_0_ts_{timestamp}.jpg")
            im1 = os.path.join(folder, f"camera_1_ts_{timestamp}.jpg")
            
            if os.path.exists(im0) and os.path.exists(im1):
                shutil.copyfile(im0, os.path.join(cam0_folder, f"{timestamp}.jpg"))
                shutil.copyfile(im1, os.path.join(cam1_folder, f"{timestamp}.jpg"))
                pairs_count += 1
            else:
                print(f"Unsynced images found for the timestamp {timestamp}")
    print(f"Copied {pairs_count} stereo pairs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rearrange synchronized images from MandEye cameras and organize into separate folders. Default naming for MandEye is cam0 = right, cam1 = left.")
    parser.add_argument("input_folder", type=str, help="Path to the input directory: continousScanning_XXXX")
    parser.add_argument("output_folder", type=str, help="Path to the output directory where images will be saved")
    
    args = parser.parse_args()
    sync_images(args.input_folder, args.output_folder)
