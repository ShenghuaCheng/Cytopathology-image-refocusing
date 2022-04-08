import os

class SampleConfig:
    '''
        sample generate config
    '''

    t_bs = 256
    t_res = 0.293
    redun_ratio = 1. / 4

    area_thre = 1. / 10
    vol_thre = 100

    pre = 'D:/paper_stage2'
    sample_save_path = os.path.join(pre, 'data_r/wsi_3D_test')
    txt_save_path = os.path.join(pre, 'data_r/txt')

    wsi_conf = [
        {
            'wsi_path': 'E:/20x_and_40x_new/20x_tiff',
            'res': 0.2428,
            # 'train_wsis': ['10140015', '10140018', '10140064', '10140071'],
            # 'val_wsis': ['10140071'],
            'test_wsis': ['10140074'],
            'postfix': '.tif',
            'sn': '3D',
            'col_thre': 20,
        },
        {
            'wsi_path': 'D:/paper_stage2/data_r/WNLO_Sfy3_pos',
            'res': 0.293,
            # 'train_wsis': ['1162026', '1162034', '1162052', '1162088'],
            # 'val_wsis': ['1162088'],
            'test_wsis': ['1167967'],
            'postfix':'.svs',
            'sn': 'our',
            'col_thre': 30
        }
    ]