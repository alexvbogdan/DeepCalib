import os
import glob

IMAGE_FILE_PATH_DISTORTED = "path to your dataset"

output_log = IMAGE_FILE_PATH_DISTORTED + "train/"
if not os.path.exists(output_log):
    os.makedirs(output_log)

output_log = IMAGE_FILE_PATH_DISTORTED + "test/"
if not os.path.exists(output_log):
    os.makedirs(output_log)

output_log = IMAGE_FILE_PATH_DISTORTED + "valid/"
if not os.path.exists(output_log):
    os.makedirs(output_log)

paths = glob.glob(IMAGE_FILE_PATH_DISTORTED + "*.jpg")
paths.sort()
for i, path in enumerate(paths[:int(len(paths)*0.8)]):
    os.rename(path,IMAGE_FILE_PATH_DISTORTED +'train/'+os.path.basename(path))
    if i%10000 == 0:
        print i, '|', int(len(paths)*0.8)
print 'train done'
for path in paths[int(len(paths)*0.8):int(len(paths)*0.9)]:
    os.rename(path,IMAGE_FILE_PATH_DISTORTED +'test/'+os.path.basename(path))
for path in paths[int(len(paths)*0.9):]:
    os.rename(path,IMAGE_FILE_PATH_DISTORTED +'valid/'+os.path.basename(path))
