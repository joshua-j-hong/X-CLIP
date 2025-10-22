# ViT-B/16
job_name="xclip_msvd_vit16"
DATA_PATH="/raid/user_data/jjhong/datasets"
CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.run --nproc_per_node=4 --master_port=29503 \
    main_xclip.py --do_train --num_thread_reader=4 \
    --epochs=3 --batch_size=320 --n_display=50 \
    --data_path ${DATA_PATH}/MSVD \
    --features_path ${DATA_PATH}/MSVD/MSVD_Videos \
    --output_dir /raid/user_data/jjhong/ckpts/ckpts3/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a log/${job_name}
