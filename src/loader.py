import cv2
import os
import math
import random
import numpy as np
import torch

data_size = 0
valid_size = 0
data_start = 0
data_array = np.arange(1)


# Load all image data before training
def load_data(args, valid=False):
    global data_size, valid_size

    filename = os.path.join(args.dataset_dir, args.index_file)
    print("From file: {}".format(filename))

    with open(filename, "r") as f:
        image_lists = f.readlines()
        random.shuffle(image_lists)
        input_path = image_lists[0].split(',')[0]
        if not os.path.exists(input_path):
            raise ValueError('input path in csv not exist: {}'.format(input_path))

    data_size = len(image_lists)
    if valid:
        valid_size = int(data_size / 20)
        data_size = data_size - valid_size
        return image_lists[:data_size], image_lists[data_size:]
    else:
        return image_lists


def get_batch_test(image_lists, batch_size, image_height, image_width):
    global data_start, data_array, color_batch, depth_batch

    if data_array.shape[0] == 1:
        data_array = np.arange(data_size)
    if data_start + batch_size > data_size:
        dlist = data_array[data_start:]
    else:
        dlist = data_array[data_start:data_start + batch_size]
        data_start = data_start + batch_size

    color_batch = torch.zeros(batch_size, 3, image_height, image_width, dtype=torch.float32)
    depth_batch = torch.zeros(batch_size, 1, image_height, image_width, dtype=torch.float32)

    batch_num = 0
    for i in dlist:
        image_names = image_lists[i].strip().split(',')
        color = cv2.imread(image_names[0], -1)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(image_names[1], -1)
        color_batch[batch_num] = torch.tensor(color.astype(np.float32)).permute(2, 0, 1)
        depth_batch[batch_num] = torch.tensor(depth.astype(np.float32))
        batch_num += 1

    return [color_batch.cuda(), depth_batch.cuda(), dlist]


def save_output(filename, output):
    image = np.squeeze(output[0].detach().cpu().numpy())
    cv2.imwrite(filename, image.astype(np.uint16))


def save_output_color(filename, output):
    image = output[0].permute(1, 2, 0).detach().cpu().numpy()
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


def crop_tensor(tensor, image_height, image_width, crop_size):
    return tensor[:, :,
           image_height // 2 - crop_size // 2:image_height // 2 + crop_size // 2,
           image_width // 2 - crop_size // 2:image_width // 2 + crop_size // 2]
