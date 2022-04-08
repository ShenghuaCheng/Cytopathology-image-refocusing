# Cervical Cytopathology Image Refocusing via Multi-scale Attention Features and Domain Normalization

## Environment
python 3.6\
tensorflow-keras 1.13.1

## Train
train_style_gray_rb.py (DNN)\
t_clear_stage1.py (RFN_s1)\
t_clear_stage2.p (RFN_s2)\
t_clear_stage1_nocycle.py (RFN_s1_no_cycle)
## Test
our_to_3D.py (DNN)\
TestRefocus.py (RFN)
## Configs
cf_style_gray_rb.py (DNN)\
cf_clear_stage1.py (RFN_s1)\
cf_clear_stage2.py (RFN_s2)

## Data
We release the cytopathology refocusing dataset at Baidu Cloud (https://pan.baidu.com/s/11VsxE8n_uX4G2Nn2jnWWwg) with and an extracting code "refo".

## Pretrained Nucleus Segmentation UNet Weights
"block380.h5" can be downloaded at the above Baidu Cloud URL

## Defocus Map Esitimation 
"Fast Defocus Map Estimation" Ding-Jie Chen, Hwann-Tzong Chen, and Long-Wen Chang ICIP 2016\
https://github.com/vllab/fast_defocus_map

## Email 
chengshen@hust.edu.cn or 905806158@qq.com

