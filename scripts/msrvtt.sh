ANNO_DIR=/var/scratch/pbagad/datasets/MSR-VTT/msrvtt_data
VIDEO_DIR=/var/scratch/pbagad/datasets/MSR-VTT/MSRVTT/videos/all/


python -w ignore -m torch.distributed.launch --nproc_per_node=3 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=72 --n_display=50 \
--train_csv ${ANNO_DIR}/MSRVTT_train.9k.csv \
--val_csv ${ANNO_DIR}/MSRVTT_JSFUSION_test.csv \
--data_path ${ANNO_DIR}/MSRVTT_data.json \
--features_path $VIDEO_DIR \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 9 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16