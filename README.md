
This software uses [UMDFaces](https://www.umdfaces.io) and [Pytorch](https://www.pytorch.org) to train deep networks for face recognition.

The steps are:
1. Generate thumbnails to train. Use `compute_aligned_images.py` for this task, point to the three batches of UMDFaces, run it three times.
2. Copy the create.py script to the val directory and run it there. This will create the missing directories in val that are required for validation to work.
3. Train. I used `python main.py --pretrained --epochs 200 --lr 0.1 --print-freq 1 /scratch2/umdfaces-thumbnails/`. You should come back 12 hours later.
4. Generate features, for example. I used the `compute_features.py` script for this task.
5. Generate plots/statistics, whatever you want, I used `run_lfw.py` for this.

