# Cervical Cytopathology Image Refocusing via Multi-scale Attention Features and Domain Normalization

Environment:python 3.6, tensorflow-keras 1.13.1

E:\Git_Hub_Res\python_project\paper_stage2
train\
train_style_gray_rb.py（域归一化）
t_clear_stage1.py（重聚焦预训练阶段）
t_clear_stage2.py（重聚焦混合训练阶段）
t_clear_stage1_nocircle.py（无circle版本，只有s1）
test\（测试模型code）
TestRefocus.py（重聚焦测试文件）
configs\（训练时所采用的配置文件）
cf_style_gray_rb.py（域归一化配置文件）
cf_clear_stage1.py （重聚焦预训练阶段配置文件）
cf_clear_stage2.py（混合训练阶段配置文件）
