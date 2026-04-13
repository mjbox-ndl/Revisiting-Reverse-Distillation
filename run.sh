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

# num_epoch=300
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

# python main.py --save_folder ./RD_aiv_stud_256_synt0114 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_stud_20250114 \
# --classes aiv_stud_0

# python main.py --save_folder ./RD_customparts_only_real_256 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --classes customparts


# num_epoch=300
# python main.py --save_folder ./RD_customparts_real_synth_256_2 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/customparts/synthetic/256/customparts \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_synth_256_mask_correction \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/customparts/synthetic/256_mask_correction/customparts \
# --classes customparts

# num_epoch=300
# python main.py --save_folder ./RD_customparts_real_100 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --real_num 100 \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_100_synth \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --real_num 100 \
# --synth_folder /media/nvme0n1p1/datasets/customparts/synthetic/256_mask_correction/customparts \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_100_synth_100 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --real_num 100 \
# --synth_num 100 \
# --synth_folder /media/nvme0n1p1/datasets/customparts/synthetic/256_mask_correction/customparts \
# --classes customparts

# num_epoch=300

# python main.py --save_folder ./RD_customparts_real_test_1 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --num_eval_epoch_start 270 \
# --num_eval_epoch 1 \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_test_2 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --num_eval_epoch_start 270 \
# --num_eval_epoch 1 \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_test_3 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --num_eval_epoch_start 270 \
# --num_eval_epoch 1 \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_synth_test_1 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --num_eval_epoch_start 270 \
# --num_eval_epoch 1 \
# --synth_folder /media/nvme0n1p1/datasets/customparts/synthetic/256_mask_correction/customparts \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_synth_test_2 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --num_eval_epoch_start 270 \
# --num_eval_epoch 1 \
# --synth_folder /media/nvme0n1p1/datasets/customparts/synthetic/256_mask_correction/customparts \
# --classes customparts

# python main.py --save_folder ./RD_customparts_real_synth_test_3 \
# --data_folder /media/nvme0n1p1/datasets/customparts/real/256 \
# --num_epoch ${num_epoch} \
# --num_eval_epoch_start 270 \
# --num_eval_epoch 1 \
# --synth_folder /media/nvme0n1p1/datasets/customparts/synthetic/256_mask_correction/customparts \
# --classes customparts

num_epoch=300

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260205 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20260205 \
# --synth_num 200 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt300_260205 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20260205 \
# --synth_num 300 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt100_260205 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20260205 \
# --synth_num 100 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260210_1 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20260210_1 \
# --synth_num 200 \
# --num_eval_epoch_start 270 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt300_260210_1 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/aiv_synthetic_screw_20260210_1 \
# --synth_num 300 \
# --num_eval_epoch_start 270 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_260209 \
# --data_folder /media/nvme0n1p1/datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --num_eval_epoch_start 270 \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260219_2840 \
# --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260219_all \
# --synth_num 200 \
# --num_eval_epoch_start 270 \
# --classes aiv_screw_3

# for bin_id in {00..09}; do
#     echo "Running bin ${bin_id}--------------------------------"
#     python main.py --save_folder ./RD_aiv_screw_256_synt200_260219_bin${bin_id} \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#         --num_epoch ${num_epoch} \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260219_bins/bin_${bin_id} \
#         --synth_num 200 \
#         --num_eval_epoch_start 270 \
#         --classes aiv_screw_3
#     echo "--------------------------------"
# done

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260219_bin09 \
#     --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#     --num_epoch ${num_epoch} \
#     --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260219_bins/bin_09 \
#     --synth_num 200 \
#     --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256 \
#     --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#     --num_epoch ${num_epoch} \
#     --classes aiv_screw_3

# for bin_id in {00..13}; do
#     echo "Running bin ${bin_id}--------------------------------"
#     python main.py --save_folder ./RD_aiv_screw_256_synt200_260223_bin${bin_id} \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#         --num_epoch ${num_epoch} \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins/bin_${bin_id} \
#         --synth_num 200 \
#         --classes aiv_screw_3
#     echo "--------------------------------"
# done

num_epoch=300
# for count in {050,100,150}; do
#     echo "Running count ${count}--------------------------------"
#     python main.py --save_folder ./RD_aiv_screw_256_synt${count}_260223_abbin \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#         --num_epoch ${num_epoch} \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_abbins/bin_${count}_00 \
#         --classes aiv_screw_3
#     echo "--------------------------------"
# done

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260223_bin01_goodonly \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#         --num_epoch ${num_epoch} \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins/bin_01_good_only \
#         --is_adding_noise_for_synth True \
#         --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_seed112 \
#     --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#     --num_epoch ${num_epoch} \
#     --seed 112 \
#     --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_seed113 \
#     --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#     --num_epoch ${num_epoch} \
#     --seed 113 \
#     --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260223_bin01_seed112 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#         --num_epoch ${num_epoch} \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins/bin_01 \
#         --seed 112 \
#         --synth_num 200 \
#         --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260223_bin01_old \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#         --num_epoch ${num_epoch} \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins/bin_01 \
#         --seed 112 \
#         --classes aiv_screw_3


# python main.py --save_folder ./RD_aiv_screw_256_synt50_260223_01_06 \
# --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins_merged/bin_01 \
# --synth_num 50 \
# --classes aiv_screw_3


# python main.py --save_folder ./RD_aiv_screw_256_synt50_260223_01_06_noise \
# --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins_merged/bin_01 \
# --synth_num 50 \
# --is_adding_noise_for_synth True \
# --classes aiv_screw_3


# python main.py --save_folder ./RD_aiv_screw_256_synt100_260223_01_06 \
# --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins_merged/bin_01 \
# --synth_num 100 \
# --classes aiv_screw_3


# python main.py --save_folder ./RD_aiv_screw_256_synt100_260223_01_06_noise \
# --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
# --num_epoch ${num_epoch} \
# --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins_merged/bin_01 \
# --synth_num 100 \
# --is_adding_noise_for_synth True \
# --classes aiv_screw_3

# python main.py --save_folder ./RD_aiv_screw_256_synt200_260223_bin01_zeroshot \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
#         --num_epoch ${num_epoch} \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260223_bins/bin_01 \
#         --real_num 0 \
#         --classes aiv_screw_3

# python main.py --save_folder ./RD_pill \
#     --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#     --num_epoch 200 \
#     --seed 111 \
#     --classes pill

# python main.py --save_folder ./RD_pill_synt200_260226 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_total \
#         --synth_num 200 \
#         --seed 111 \
#         --classes pill

# python main.py --save_folder ./RD_pill_synt200_260226_bin00 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_00 \
#         --seed 111 \
#         --classes pill

# python main.py --save_folder ./RD_pill_synt200_260226_bin01 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_01 \
#         --seed 111 \
#         --classes pill
        
# python main.py --save_folder ./RD_pill_synt200_260226_bin02 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_02 \
#         --seed 111 \
#         --classes pill

# python main.py --save_folder ./RD_pill_synt200_260226_bin02_256 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_02_256 \
#         --seed 111 \
#         --classes pill

# for bin_id in {00..04}; do
#     echo "Running bin ${bin_id}--------------------------------"
#     python main.py --save_folder ./RD_pill_synt200_260302_bin${bin_id} \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260302_bins/bin_${bin_id} \
#         --seed 111 \
#         --classes pill
#     echo "--------------------------------"
# done


# python main.py --save_folder ./RD_pill_synt200_260227_bin_02_256_goodonly \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_02_256_good \
#         --classes pill

# python main.py --save_folder ./RD_pill_synt50_260227_bin_02_256 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_02_256_50 \
#         --classes pill

# python main.py --save_folder ./RD_pill_synt100_260227_bin_02_256 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_02_256_100 \
#         --classes pill

# python main.py --save_folder ./RD_pill_synt150_260227_bin_02_256 \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_02_256_150 \
#         --classes pill

# python main.py --save_folder ./RD_screw_synt200_260227_total_goodonly \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 280 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/screw_20260227_total_good \
#         --classes screw

# python main.py --save_folder ./RD_screw_synt50_260227_total \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 280 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/screw_20260227_bins/bin_050_00 \
#         --classes screw

# python main.py --save_folder ./RD_screw_synt100_260227_total \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 280 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/screw_20260227_bins/bin_100_00 \
#         --classes screw

# python main.py --save_folder ./RD_screw_synt150_260227_total \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 280 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/screw_20260227_bins/bin_150_00 \
#         --classes screw



# python main.py --save_folder ./RD_pill_synt200_260227_bin_02_256_zeroshot \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 200 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/pill_20260227_bins/bin_02_256 \
#         --real_num 0 \
#         --classes pill

# python main.py --save_folder ./RD_screw_synt200_260227_total_zeroshot \
#         --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/mvtec/ \
#         --num_epoch 280 \
#         --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/screw_20260227_total \
#         --real_num 0 \
#         --classes screw

# next have to run for screw & pill num_of_synth [50, 100, 150]


python main.py --save_folder ./RD_aiv_screw_256_synt50_260309_bin_01 \
        --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
        --num_epoch 300 \
        --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260309/bin_01_050_00 \
        --classes aiv_screw_3

python main.py --save_folder ./RD_aiv_screw_256_synt100_260309_bin_01 \
        --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
        --num_epoch 300 \
        --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260309/bin_01_100_00 \
        --classes aiv_screw_3

python main.py --save_folder ./RD_aiv_screw_256_synt150_260309_bin_01 \
        --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
        --num_epoch 300 \
        --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260309/bin_01_150_00 \
        --classes aiv_screw_3
        
python main.py --save_folder ./RD_aiv_screw_256_synt200_260309_bin_01 \
        --data_folder /media/nvme0n1p1/datasets/anomaly_datasets/aiv_mvtec/ \
        --num_epoch 300 \
        --synth_folder /media/nvme0n1p1/datasets/anomaly_datasets/synthetic_topo_2026/aiv_synthetic_screw_20260309/bin_01 \
        --classes aiv_screw_3
        