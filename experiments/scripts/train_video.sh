python train.py --model torchvision/video/r3d_18 --sched cosine --epoch 30 --batch-size 64 --warmup-epochs 5 --cooldown-epochs 0 --opt adamw --lr 1e-3 --weight-decay 0.01 --dataset hmdb --num-classes 51 --num-workers 64 --compile --kas-inference-time-limit 1200 --input-size 3 112 112

# train_data = HMDB51(
#             root=os.path.join("/cephfs/suzhengyuan/data", "hmdb/videos"), 
#             annotation_path=os.path.join("/cephfs/suzhengyuan/data", "hmdb/annotations"), 
#             frames_per_clip=8, 
#             step_between_clips=8, 
#             num_workers=16,
#             train=True
#         )