# python main.py --data_folder /media/nvme0n1p1/datasets/mvtec --classes transistor \
# --save_folder ./RD_transistor_100 --synth_folder /media/nvme0n1p1/datasets/transitor_20241223

# python main.py --data_folder /media/nvme0n1p1/datasets/mvtec --classes transistor \
# --save_folder ./RD_transistor --classes transistor

# python main.py --save_folder ./RD_aiv_screw_88 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --classes aiv_screw

# python main.py --save_folder ./RD_aiv_screw_88_1_synthetic \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ --synth_folder /media/nvme0n1p1/datasets/aiv_syntetic_20241230 \
# --classes aiv_screw 

# mv /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw_1
# mv /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw_2 /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw

# python main.py --save_folder ./RD_aiv_screw_88_2 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --classes aiv_screw

# python main.py --save_folder ./RD_aiv_screw_88_2_synthetic2 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ --synth_folder /media/nvme0n1p1/datasets/aiv_syntetic_20241230 \
# --classes aiv_screw 

num_epoch=300
# python main.py --save_folder ./RD_aiv_screw_88_3_256 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_4_256_2 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/  --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_20241231_2_256 \
# --num_epoch ${num_epoch} --synth_num -1 \
# --classes aiv_screw_4

# python main.py --save_folder ./RD_aiv_screw_4_256_merged2 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/  --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_merged_2_256 \
# --num_epoch ${num_epoch} --synth_num -1 \
# --classes aiv_screw_4

# python main.py --save_folder ./RD_aiv_screw_88_3_synthetic2 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ --synth_folder /media/nvme0n1p1/datasets/aiv_syntetic_20241230 \
# --classes aiv_screw 

# mv /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw_3
# mv /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw_2 /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw

# python main.py --save_folder ./RD_aiv_screw_88_2_synthetic3 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_20241231_2 \
# --classes aiv_screw 

# mv /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw_2
# mv /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw_1 /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw

# python main.py --save_folder ./RD_aiv_screw_88_1_synthetic3 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_20241231_2 \
# --classes aiv_screw 


# mv /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw /media/nvme0n1p1/datasets/aiv_mvtec/aiv_screw_1

# python main.py --save_folder ./RD_aiv_screw_3_256_aug_20 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --brightness_range 0.8 1.2 --contrast_range 0.8 1.2 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_3_256_no_aug \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_3_256_synt0110 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20250110 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_stud_256 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --classes aiv_stud_0

# python main.py --save_folder ./RD_aiv_screw_3_256_synt0110_jb \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20250110_jb \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_3_256_synt0110_jb_noise \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20250110_jb \
# --classes aiv_screw_3


# python main.py --save_folder ./RD_aiv_screw_3_256_synt0113 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20250113 \
# --classes aiv_screw_3

python main.py --save_folder ./RD_aiv_stud_256_synt0114 \
--data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
--num_epoch ${num_epoch} \
--synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_stud_20250114 \
--classes aiv_stud_0


