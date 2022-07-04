DATA_PATH=/var/scratch/pbagad/datasets/DiDeMo
python -W ignore -m torch.distributed.launch --nproc_per_node=3 \
--master_port 9998 main_task_retrieval.py --do_eval --num_thread_reader=2 \
--epochs=5 --batch_size=9 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos \
--output_dir ckpts/ckpt_didemo_retrieval_tightType \
--lr 1e-4 --max_words 64 --max_frames 32 --batch_size_val 3 \
--datatype didemo --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header tightTransf \
--pretrained_clip_name ViT-B/16 \
--init_model ckpts/ckpt_didemo_retrieval_tightType/pytorch_model.bin.0 \
--eval_frame_order 2 
