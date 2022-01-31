This repository provides the code in our ECCV paper 

"
[Learning to Predict Salient Faces: A Novel Visual-Audio Saliency Model](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650409.pdf)
"

## Abstract
Recently, video streams have occupied a large proportion of
Internet traffic, most of which contain human faces. Hence, it is necessary to predict saliency on multiple-face videos, which can provide
attention cues for many content based applications. However, most of
multiple-face saliency prediction works only consider visual information
and ignore audio, which is not consistent with the naturalistic scenarios. Several behavioral studies have established that sound influences
human attention, especially during the speech turn-taking in multipleface videos. In this paper, we thoroughly investigate such influences by
establishing a large-scale eye-tracking database of Multiple-face Video in
Visual-Audio condition (MVVA). Inspired by the findings of our investigation, we propose a novel multi-modal video saliency model consisting
of three branches: visual, audio and face. The visual branch takes the
RGB frames as the input and encodes them into visual feature maps. The audio and face branches encode the audio signal and multiple
cropped faces, respectively. A fusion module is introduced to integrate
the information from three modalities, and to generate the final saliency
map. Experimental results show that the proposed method outperforms
11 state-of-the-art saliency prediction works. It performs closer to human
multi-modal attention.

## Network
![](https://github.com/MinglangQiao/MVVA-Database/blob/master/fig/talking/all_video-2.jpg)

## Requirements
* pytorch
* opencv
* 

## Inference
Download the pretrained model


## 

