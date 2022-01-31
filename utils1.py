from tkinter import ttk
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import models
import librosa
import librosa.display
# import torchaudio
import copy
import scipy.ndimage as ndimage

from config import *
# from visualization import *

def load_one_video_audio(path, final_frame, fps, input_dim, hop_len=20, hop_sample_num=512, 
    audio_sample_rate=22050, raw_h=720, raw_w=1280):
    """
    0813 update

    Parameters:
    -----------
    hop_len: the len of audio for one visual frame, 20 * 23 ms = 460 ms

    Reference:
    ----------
    human audio frequency: https://zh.wikipedia.org/wiki/%E8%AF%AD%E9%9F%B3%E9%A2%91%E7%8E%87

    the output len: https://blog.csdn.net/c2c2c2aa/article/details/81583973

    librosa.feature.melspectrogram: https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
    """
    # print(">>>> path: ", path)
    # tt

    y, sr = librosa.load(path)  # y: the audio data;  sr: sample rate.  sr(default)=22050
    # print(">>>> y: ", np.shape(y))
    # tt

    melspec = librosa.feature.melspectrogram(y, sr)
    logmelspec = librosa.power_to_db(melspec, ref=np.max)

    logmelspec = logmelspec - np.min(logmelspec)
    logmelspec = logmelspec/np.max(logmelspec) * 255

    ### convert to frmaes 
    one_video_audio = []
    one_frame_time = 1000.0/fps ## ms
    one_hop_time = hop_sample_num * 1000/audio_sample_rate # ms
    mel_spectr_len = np.shape(logmelspec)[1]

    for i_frame in range(final_frame):
        frame_point = int(one_frame_time * i_frame / one_hop_time)
        one_frame_start = frame_point - int(hop_len/2)
        one_frame_end = frame_point + int(hop_len/2)
        if one_frame_start < 0:
            off_set = 0 - one_frame_start
        elif one_frame_end > mel_spectr_len:
            off_set = mel_spectr_len - one_frame_end
        else:
            off_set = 0
        one_frame_start += off_set ## has to ensue (end-start) = 112
        one_frame_end += off_set
        one_frame_audio = logmelspec[:, one_frame_start:one_frame_end]

        one_frame_audio = cv2.resize(one_frame_audio, (input_dim, input_dim))
        one_video_audio.append(one_frame_audio)

    print(">>>> load one video audio: {} {} {}".format(raw_w, raw_h, np.shape(one_video_audio)))
    # t
    return one_video_audio

def load_one_video_frames(read_one_video_path):
    
    one_video_all_frames = []
    cap = cv2.VideoCapture(read_one_video_path)
    success, image = cap.read()
    count = 1
    one_video_all_frames.append(image)
    while success:
        # cv2.imwrite(save_video_path + "/%03d.png"%count, image)
        success, image = cap.read()
        count += 1
        if len(np.shape(image)) == 3:
            one_video_all_frames.append(image)
            # print(np.shape(image))
        # cv2.imshow("t1", image)
        # cv2.waitKey()
    actal_video_frames = np.shape(one_video_all_frames)[0]
    print(">>>> load one video all frames done!, data_shape: {}\n".format(
        np.shape(one_video_all_frames)))

    return one_video_all_frames, actal_video_frames

def load_one_video_gmm(path):
    
    one_video_all_frames = []
    frame_list = os.listdir(path)
    # print(frame_list[:10])
    # t
    for i_frame in range(len(frame_list)):
        frame_path = path + "/%d.jpg"%(i_frame)
        one_frame_data = cv2.imread(frame_path, 0) ## 0 means read as gray format
        one_video_all_frames.append(one_frame_data)
        
    hmap_frame_num = np.shape(one_video_all_frames)[0]
    print(">>>> load one video gmm maps done!, data_shape: {}\n".format(
        np.shape(one_video_all_frames)))
    # t
    return one_video_all_frames, hmap_frame_num

def load_one_batch_image(one_video_frames, i_batch, train_batch_size, input_dim, frame_offset=0, 
    final_frame=None):
    """
    frame_offset is for load image of optical flow

    """
    one_batch_image = []
    one_batch_raw_image = []
    one_batch_start = frame_offset + i_batch * train_batch_size
    one_batch_end = frame_offset + (i_batch + 1) * train_batch_size
    if not final_frame == None:
        if one_batch_end > final_frame:
            return -1, -1

    for i_frame in range(one_batch_start, one_batch_end):
        one_frame = one_video_frames[i_frame]

        raw_h, raw_w, c = np.shape(one_frame)
        one_frame = cv2.resize(one_frame, (input_dim, input_dim))


        inputimage = cv2.cvtColor(one_frame, cv2.COLOR_BGR2RGB) ## (720, 1280, 3)
        ## put the channnel dim first
        inputimage = np.transpose(inputimage, (2, 0, 1))
        one_batch_raw_image.append(inputimage)
        inputimage = inputimage - MEAN_VALUE
        inputimage = inputimage.astype(np.dtype(np.float32))
        one_batch_image.append(inputimage)
        # print(np.shape(one_frame))
        # t
        # cv2.imshow("1", one_frame)
        # cv2.waitKey()
    # print(">>>> load one batch images done!, data_shape: {}\n".format(
    #     np.shape(one_batch_image)))
    one_batch_image = torch.from_numpy(np.array(one_batch_image))
    return one_batch_image, one_batch_raw_image

def load_one_batch_audio_melspectr(one_video_audio, i_batch, train_batch_size, frame_offset=0, 
    final_frame=None, one_visual_frame_3D_frames=16, debug=False):
    """

    output dimension: [batch, channel, frames, h, w]

    one frame len: 16 * 23 ms = 528 ms

    for each visual frame, we use frame_T number of point as the corresponding input

    each point is one hop lenght, about 23 ms
    """
    one_batch_audio = []
    
    one_batch_start = frame_offset + i_batch * train_batch_size
    one_batch_end = frame_offset + (i_batch + 1) * train_batch_size

    if not final_frame == None:
        if one_batch_end > final_frame:
            return -1, -1

    for i_frame in range(one_batch_start, one_batch_end):

        one_frame_start = i_frame - int(one_visual_frame_3D_frames/2)
        one_frame_end = i_frame + int(one_visual_frame_3D_frames/2)
        
        if one_frame_start < 0:
            off_set = 0 - one_frame_start
        elif one_frame_end > final_frame:
            off_set = final_frame - one_frame_end
        else:
            off_set = 0
        one_frame_start += off_set ## has to ensue (end-start) = 112
        one_frame_end += off_set
        # print(">>>> i_batch: {} i_frame: {}, one_frame_s: {}, end: {}".format(i_batch, i_frame, 
        #     one_frame_start, one_frame_end))
        one_frame_audio = []
        for i_step in range(one_frame_start, one_frame_end):
            one_step_audio = one_video_audio[i_step]
            one_step_audio = np.expand_dims(one_step_audio, axis=0)
            one_frame_audio.append(one_step_audio)
            # print(np.shape(one_step_audio))
            # t
        one_batch_audio.append(one_frame_audio)

    one_batch_audio = np.array(one_batch_audio).transpose(0, 2, 1, 3, 4)
    one_batch_audio = torch.from_numpy(np.array(one_batch_audio)) # (64, 113)
    # print(">>>> shape: {}, max: {}".format(np.shape(one_batch_audio), np.shape(one_batch_audio[0]))) # torch.Size([12, 1, 16, 112, 112])
    # t

    return one_batch_audio


def load_one_batch_gmm(one_video_frames, i_batch, train_batch_size, output_dim,
    frame_offset=0, final_frame=None):
    """
    0809 update by ml

    there is a problem that, each batch has no overlayped

    """
    # print(">>>> d1 {}".format(np.shape(one_video_frame)))
    # t
    one_batch_hmap = []
    one_batch_start = frame_offset + i_batch * train_batch_size
    one_batch_end = frame_offset + (i_batch + 1) * train_batch_size
    if not final_frame == None:
        if one_batch_end > final_frame:
            return -1, -1
    # print(">>>> ", one_batch_end, final_frame, final_frame)

    for i_frame in range(one_batch_start, one_batch_end):
        one_frame = one_video_frames[i_frame]
        saliencyimage = one_frame # cv2.resize(one_frame, (output_dim, output_dim))
        # print(">>>> d2: {}".format(np.shape(saliencyimage)))
        # t
        saliencyimage = saliencyimage - np.min(saliencyimage)
        if np.max(saliencyimage) > 0:
            saliencyimage = saliencyimage/np.max(saliencyimage) * 255 - 14.9 ## 14.9 is the ave value of all training gmms
        else:
            print(">>>>>>>>> np.max(gmm) <= 0")
        # print(">>>> {}".format(np.max(saliencyimage)))
        # t
        ## put the channnel dim first
        saliencyimage = saliencyimage.astype(np.dtype(np.float32))
        saliencyimage = np.expand_dims(saliencyimage, axis=0) # 1 * 37 * 33
        one_batch_hmap.append(saliencyimage) # (8, 3, 448, 448)
        # print("np.shape(saliencyimage), np.shape(one_batch_hmap): ", np.shape(saliencyimage), np.shape(one_batch_hmap))

    one_batch_hmap = torch.from_numpy(np.array(one_batch_hmap))
    # print(">>>> load one batch hmaps done!, data_shape: {}\n".format(
    #       np.shape(one_batch_hmap)))
    # t
    return one_batch_hmap

def process_output(outputs_map, image_size_W, image_size_H):
    
    batch_size = len(outputs_map)
    saliency_map = []
    for i_image in range(batch_size):
        sal_map = outputs_map[i_image, :, :]
        sal_map = sal_map - np.min(sal_map)
        sal_map = sal_map / np.max(sal_map) * 255
        sal_map = cv2.resize(sal_map, (image_size_W, image_size_H)) ## INTER_LINEAR interpolation
        saliency_map.append(sal_map)

    return np.array(saliency_map)