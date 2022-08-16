import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import face_recognition
import multiprocessing
import itertools
import warnings
import shutil
warnings.filterwarnings('ignore')
import sys
import argparse

def detect_template_face_v2(image_path, 
                            template_image,
                            save_dir,
                            model_name='hog',
                            upsample_count=1
                           ):
    image = cv2.imread(image_path)
    face_locations = face_recognition.face_locations(image, 
                                                     number_of_times_to_upsample = upsample_count,
                                                     model='hog')
    face_encodings = face_recognition.face_encodings(image, face_locations)
    template_embedding = face_recognition.face_encodings(template_image)
    if len(template_embedding) == 0:
        raise Exception('No face Found in template')
    template_embedding = template_embedding[0]
    for encoding in face_encodings:
        results = face_recognition.compare_faces([template_embedding],encoding)
        if results[0]:
            shutil.copy(image_path, save_dir)
            print(f'{os.path.basename(image_path)} saved')
            return True
    return False 

def detect_face_wrapper(image_path_list,
                        template_path,
                        save_dir , 
                        cpu_count=-1,
                        model='hog',
                        upsample_count=1):
    if cpu_count == -1:
        processes = None
    else:
        processes = cpu_count

    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)
    template_image = cv2.imread(template_path)
    function_parameters = zip(
        image_path_list,
        itertools.repeat(template_image),
        itertools.repeat(save_dir),
        itertools.repeat(model),
        itertools.repeat(upsample_count),
    )

    results = pool.starmap(detect_template_face_v2, function_parameters)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',type=str, help='path to directory where images are stored')
    parser.add_argument('--template_path',type=str, help='template path of face to search for')
    parser.add_argument('--save_dir', type=str, help='path to save images')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    path_list = [ os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)]
    detect_face_wrapper(path_list[:50], template_path=args.template_path, save_dir=args.save_dir) 
