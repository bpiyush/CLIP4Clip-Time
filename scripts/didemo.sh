DATA_PATH=/var/scratch/pbagad/datasets/DiDeMo
python -W ignore -m torch.distributed.launch --nproc_per_node=3 \
--master_port 9998 main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=9 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos \
--output_dir ckpts/ckpt_didemo_retrieval_looseType \
--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 9 \
--datatype didemo --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16