from tkinter import ttk
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torchviz import make_dot, make_dot_from_trace
import torch.nn.functional as F
from torchvision import models

import numpy as np

from model import visual_audio_face_modal_combine_net
from utils1 import *
from tqdm import trange

import matplotlib.pyplot as plt
import cv2

def test():
    
    model_path = "./data/17_i_video_120_host_207_station_0905_visual_audio_face_modal_cc_0.690_kl_0.8210.pkl"
    video_path = "./data/"
    test_gmm_path = "./data/resized_gmm/"
    audio_path = './data/'
    input_dim = 256
    save_hmap_path = "./data/result/"
    Test_Batch_Size = 12
    frame_skip = 5
    output_dim = 32
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_model = visual_audio_face_modal_combine_net()

    current_model.load_state_dict(torch.load(model_path))
    current_model.to(device)

    current_model.eval()
    with torch.no_grad():
        # tt

        # video_name, fps, frame_num0, video_width, video_height = all_session_video_final[int(video_name_short)]

        video_name = '155.avi'
        video_name_short = video_name[:-4]
        fps = 24

        one_video_gmm, gmm_frame_num = load_one_video_gmm(test_gmm_path + video_name_short)

        one_video_frames, actal_video_frames = load_one_video_frames(video_path + video_name)

        raw_h, raw_w, _ = np.shape(one_video_frames[0])

        one_video_audio = load_one_video_audio(audio_path + video_name[:-4] + ".wav", 
                            actal_video_frames, fps, int(input_dim/4), raw_h=raw_h, raw_w=raw_w)

        frame_list = [actal_video_frames, gmm_frame_num]
        final_frame = int(np.min(frame_list))
        one_video_batch_num = int(final_frame/Test_Batch_Size)

        for i_batch in trange(one_video_batch_num):
            one_batch_frame, one_batch_raw_image = load_one_batch_image(one_video_frames, i_batch, 
                Test_Batch_Size, input_dim, frame_offset=0, final_frame=final_frame)
            one_batch_frame_offset_5, one_batch_raw_image_offset_5 = load_one_batch_image(one_video_frames, i_batch, 
                Test_Batch_Size, input_dim, frame_offset=frame_skip, final_frame=final_frame)
            if len(np.shape(one_batch_frame_offset_5)) == 0:
                continue
            
            one_batch_audio_offset_5 = load_one_batch_audio_melspectr(one_video_audio, i_batch, 
                Test_Batch_Size, final_frame=final_frame, frame_offset=frame_skip)

            one_batch_gmm_offset_5 = load_one_batch_gmm(one_video_gmm, i_batch, 
                Test_Batch_Size, output_dim, frame_offset=frame_skip, final_frame=final_frame)

            one_batch_frame = one_batch_frame.to(device)
            one_batch_frame_offset_5 = one_batch_frame_offset_5.to(device)
            one_batch_audio_offset_5 = one_batch_audio_offset_5.to(device)
            one_batch_gmm_offset_5 = one_batch_gmm_offset_5.to(device)

            outputs = current_model(one_batch_frame, one_batch_frame_offset_5, 
                                    one_batch_audio_offset_5, 
                                    one_batch_gmm_offset_5, 
                                    i_batch).to(device).squeeze().type(torch.FloatTensor)
            
            print(">>>> outputs: ", np.shape(outputs))
            
            outputs_map = F.softmax(outputs, 1)
            outputs_map = outputs_map.view(Test_Batch_Size, output_dim, output_dim).detach().numpy()
            outputs_saliency_map = process_output(outputs_map, raw_w, raw_h)

            cv2.imwrite('t1.jpg', outputs_saliency_map[0])
            tt


if __name__ == '__main__':
    test()