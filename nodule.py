from operator import itemgetter
import math
import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn.metrics

cur_dir = os.path.abspath('.')
root_dir = os.path.dirname(cur_dir)

sys.path.append(os.path.join(cur_dir, 'Mask_RCNN'))
import mrcnn.config
import mrcnn.model
import mrcnn.parallel_model
import mrcnn.visualize
import mrcnn.utils

# Directory to save logs and trained model
model_dir = os.path.join(cur_dir, "logs")

# Local path to trained weights file
coco_model_path = os.path.join(cur_dir, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(coco_model_path):
    utils.download_trained_weights(coco_model_path)


Config = mrcnn.config.Config



def normalize(image, ww=1500, wl=-400):
    """normalize ct scan value to 0~1 according to given window-width and window-location"""

    low, high = wl - ww / 2, wl + ww / 2
    image = np.clip(image, low, high)
    image = (image - low) / ww
    return image


class NoduleDataset(mrcnn.utils.Dataset):

    def __init__(self, df_record, class_mapper, val_count=800, *args, **kw):
        super().__init__(*args, **kw)

        self.class_mapper = class_mapper
        self.df_all = df_record
        if 'subset' not in self.df_all.columns:
            # split train data and validation data
            self.df_all['subset'] = 'train'
            val_idx = np.random.choice(self.df_all.shape[0], val_count, replace=False)
            self.df_all.loc[val_idx, 'subset'] = 'val'

            # normalize label to continuous integers
            labels = sorted(self.df_all['label'].unique().tolist())
            mapper = dict(zip(labels, range(1, len(labels) + 1)))
            self.df_all['ilabel'] = self.df_all['label'].apply(lambda lb: mapper[lb])

    def load_dataset(self, subset):
        """Generate the requested number of synthetic images."""
        assert subset in ('train', 'val')

        # Add classes
        for id, name in self.class_mapper.items():
            self.add_class('nodule', id, name)

        # Add images
        self.df = self.df_all[self.df_all['subset'] == subset]

        image_ids = set()
        for row in self.df.itertuples():
            image_id = (row.seriesuid, row.coordZ)
            path = os.path.join(cur_dir, 'data', 'train', '{}_{}.npy'.format(row.seriesuid, row.coordZ))
            if image_id in image_ids:
                continue
            self.add_image("nodule", image_id=image_id, path=path)
            image_ids.add(image_id)

    def load_image_old(self, image_id, ww=1500, wl=-400):
        seriesuid, zindex = self.image_info[image_id]['id']
        mhd_file = os.path.join(root_dir, 'train', '{}.mhd'.format(seriesuid))
        itk_image = sitk.ReadImage(mhd_file)
        image3d = sitk.GetArrayViewFromImage(itk_image)
        image = image3d[zindex]
        # image = (normalize(image, ww, wl) * ww).astype(np.int32)
        image = normalize(image, ww, wl)
        image = image[..., np.newaxis]
        return image

    def load_image(self, image_id, ww=1500, wl=-400):
        image = np.load(self.image_info[image_id]['path'])
        image = (normalize(image, ww, wl) * ww).astype(np.int32)
        # image = normalize(image, ww, wl)
        image = image[..., np.newaxis]
        return image

    def load_bbox(self, image_id):
        seriesuid, zindex = self.image_info[image_id]['id']
        df_this = self.df[(self.df['seriesuid'] == seriesuid) & (self.df['coordZ'] == zindex)]
        bbox = np.zeros((df_this.shape[0], 4))
        for i, row in enumerate(df_this.itertuples()):
            x, y, width, height = row.coordX, row.coordY, row.diameterX, row.diameterY
            bbox[i] = [y - height/2, x - width/2, y + height/2, x + width/2]
        class_ids = df_this['ilabel'].values
        return bbox, class_ids


def load_datasets(df_record, class_mapper, val_split=0.08):
    val_count = round(df_record.shape[0] * val_split)

    dataset_train = NoduleDataset(df_record=df_record, class_mapper=class_mapper, val_count=val_count)
    dataset_train.load_dataset('train')
    dataset_train.prepare()

    dataset_val = NoduleDataset(df_record=df_record, class_mapper=class_mapper)
    dataset_val.load_dataset('val')
    dataset_val.prepare()

    return dataset_train, dataset_val


def train(config, dataset_train, dataset_val, epochs, tune_epochs, init_with='coco'):
    # Create model in training mode
    model = mrcnn.model.MaskRCNN(mode="training", config=config, model_dir=model_dir)

    # Which weights to start with? imagenet, coco, or last
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(
            coco_model_path, by_name=True,
            exclude=[
                'conv1', 'mrcnn_class_logits', 'mrcnn_bbox_fc',
                'mrcnn_bbox', 'mrcnn_mask'])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(
        dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=epochs,
        layers=r'(conv1)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(
        dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=tune_epochs,
        layers='all')

    return model


def load_model(config, mode='inference', model_path=None):
    # Recreate the model in inference mode
    model = mrcnn.model.MaskRCNN(
        mode=mode,
        config=config,
        model_dir=model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if model_path is None:
        model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model


def inference(model, dataset, image_ids, config, detect_threshold=0, verbose=0):
    # Test on a random image
    ncol = 2
    nrow = len(image_ids)
    ax_size = 16 / ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ax_size * ncol, ax_size * nrow))
    fig.tight_layout()

    for i, image_id in enumerate(image_ids):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        ax1.axis('off')
        ax2.axis('off')

        original_image, image_meta, gt_class_id, gt_bbox = mrcnn.model.load_image_gt(
            dataset, config, image_id, use_mini_mask=False)

        mrcnn.visualize.display_instances(
            original_image, gt_bbox, gt_class_id, dataset.class_names,
            ax=ax1, cmap=plt.cm.gray)

        results = model.detect([original_image], verbose=verbose)
        r = results[0]
        idx = r['scores'] > detect_threshold
        r['rois'] = r['rois'][idx]
        r['scores'] = r['scores'][idx]
        r['class_ids'] = r['class_ids'][idx]
        mrcnn.visualize.display_instances(
            original_image, r['rois'], r['class_ids'], dataset.class_names,
            r['scores'], ax=ax2, cmap=plt.cm.gray)


def predict(model, seriesuid, datapath='train/', ww=1500, wl=-400, detect_threshold=0):
    # df = pd.DataFrame(columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])

    mhd_file = os.path.join(root_dir, datapath, '{}.mhd'.format(seriesuid))
    itk_image = sitk.ReadImage(mhd_file)
    image3d = sitk.GetArrayViewFromImage(itk_image)

    columns = ['seriesuid', 'z', 'y1', 'x1', 'y2', 'x2', 'label', 'score']
    row_list = []
    for i in range(image3d.shape[0]):
        image = image3d[i]
        image = (normalize(image, ww, wl) * ww).astype(np.int32)
        image = image[..., np.newaxis]
        results = model.detect([image], verbose=0)
        r = results[0]
        if len(r['class_ids']) == 0:
            continue
        idx = r['scores'] > detect_threshold
        r['rois'] = r['rois'][idx]
        r['scores'] = r['scores'][idx]
        r['class_ids'] = r['class_ids'][idx]
        for j in range(len(r['class_ids'])):
            row_list.append([seriesuid, i, *r['rois'][j], r['class_ids'][j], r['scores'][j]])
    df = pd.DataFrame(row_list, columns=columns)
    return df



def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def group_sequence(lst, max_gap=0):
    lst = sorted(list(lst))
    groups = []
    group = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] <= max_gap + 1:
            group.append(lst[i])
        else:
            groups.append(group)
            group = [lst[i]]
    if group:
        groups.append(group)
    return groups

def softmax(arr):
    tmp = np.power(np.e, arr)
    return tmp / sum(tmp)

def my_tanh(x, base=np.e):
    a, b = np.power(base, x), np.power(base, -x)
    return (a - b) / (a + b)

def boxes_to_3d(df, iou_threashold=0.3, max_zindex_gap=1):
    """input dataframe with columns: ['seriesuid', 'z', 'y1', 'x1', 'y2', 'x2', 'label', 'score'],
    produced by function `predict`.
    """
    df = df.sort_values(by=['seriesuid', 'label', 'z'])
    df['box_area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])

    groups = []
    for _, sub_df in df.groupby(['seriesuid', 'label']):
        # 将 boxes 按照病灶类型分组
        left_df = sub_df
        while not left_df.empty:
            # 然后根据 IoU 大小判定重叠程度并分组
            box = left_df.iloc[0][['y1', 'x1', 'y2', 'x2']].values
            box_area = left_df.iloc[0]['box_area']
            boxes = left_df[['y1', 'x1', 'y2', 'x2']].values
            boxes_area = left_df['box_area'].values
            ious = compute_iou(box, boxes, box_area, boxes_area)

            idx = (ious > iou_threashold)
            iou_group = left_df[idx]
            left_df = left_df[~idx]

            # 重叠程度高并不一定指向同一个病灶，还要求 zindex “连续”
            zindex = iou_group['z'].values.tolist()
            grouped_zindex = group_sequence(zindex, max_zindex_gap)
            start = 0
            for gz in grouped_zindex:
                # 每个 group 中的所有 box 指示同一个病灶
                groups.append(iou_group.iloc[start:start+len(gz)])
                start = start + len(gz)

    row_list = []
    for mini_df in groups:
        # 病灶的 x, y, z 坐标为所有 boxes 中心坐标的加权平均
        # 病灶的整体 probability 需要基于所有 boxes 的 probability 进行一定的换算，这个换算应该满足以下要求：
        # 1. 换算后的值在 0~1 之间
        # 2. boxes 越多，整体 probability 越大
        # 3. 每个 box 的 probability 越大，整体 probability 越大
        # PS: 当病灶在同一个 zindex 中存在多个 box 时，probability 不叠加，而是取其中的最大值
        scores = mini_df['score'].values
        weights = softmax(scores)
        xs = (mini_df['x1'] + mini_df['x2']) / 2
        ys = (mini_df['y1'] + mini_df['y2']) / 2
        zs = mini_df['z']
        x = (xs * weights).sum()
        y = (ys * weights).sum()
        z = (zs * weights).sum()
        # probability = (probs * weights).sum()
        probs = mini_df.groupby(['z'])['score'].max().values
        # clip = lambda x: np.clip(x, 0, 6) / 6
        clip = lambda x: my_tanh(x, base=1.5)
        transform = lambda x: x
        probability = transform(clip(probs.sum()))

        seriesuid = int(mini_df.iloc[0]['seriesuid'])
        label = int(mini_df.iloc[0]['label'])
        row_list.append([seriesuid, x, y, z, label, probability])

    df = pd.DataFrame(row_list, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])
    df = df.sort_values(['seriesuid', 'probability'], ascending=[True, False]).reset_index(drop=True)
    return df


def world_to_voxel(df, datapath='train/'):
    df = df.copy()
    for uid in df['seriesuid'].unique():
        idx = df['seriesuid'] == uid
        mhd_file = os.path.join(root_dir, datapath, '{}.mhd'.format(uid))
        itk_image = sitk.ReadImage(mhd_file)
        origin = np.array(itk_image.GetOrigin())
        spacing = np.array(itk_image.GetSpacing())
        df.loc[idx, ['coordX', 'coordY', 'coordZ']] = ((df.loc[idx, ['coordX', 'coordY', 'coordZ']] - origin) / spacing).round()
        df.loc[idx, ['diameterX', 'diameterY', 'diameterZ']] = (df.loc[idx, ['diameterX', 'diameterY', 'diameterZ']] / spacing).round()
    return df


def voxel_to_world(df, datapath='train/'):
    df = df.copy()
    for uid in df['seriesuid'].unique():
        idx = df['seriesuid'] == uid
        mhd_file = os.path.join(root_dir, datapath, '{}.mhd'.format(uid))
        itk_image = sitk.ReadImage(mhd_file)
        origin = np.array(itk_image.GetOrigin())
        spacing = np.array(itk_image.GetSpacing())
        df.loc[idx, ['coordX', 'coordY', 'coordZ']] = df.loc[idx, ['coordX', 'coordY', 'coordZ']] * spacing + origin
        # df.loc[idx, ['diameterX', 'diameterY', 'diameterZ']] = (df.loc[idx, ['diameterX', 'diameterY', 'diameterZ']] / spacing).round()
    return df


def get_froc(df_gt, df_predict):
    frocs = {}
    df_predict = df_predict.copy()
    df_predict['gt'] = 0
    for label in df_gt['label'].unique():
        idx_lb = (df_gt['label'] == label)
        df_predict_lb = df_predict[df_predict['class'] == label]

        for i, row in enumerate(df_predict_lb.itertuples()):
            idx_uid = (df_gt['seriesuid'] == row.seriesuid)
            for row_gt in df_gt[idx_lb & idx_uid].itertuples():
                if all([
                        abs(row.coordX - row_gt.coordX) <= row_gt.diameterX / 2,
                        abs(row.coordY - row_gt.coordY) <= row_gt.diameterY / 2,
                        abs(row.coordZ - row_gt.coordZ) <= row_gt.diameterZ / 2,
                    ]):
                    df_predict_lb.loc[:, 'gt'].iloc[i] = 1

        gt_list = df_predict_lb['gt'].values.tolist()
        prob_list = df_predict_lb['probability'].values.tolist()
        # 补充未预测到的病灶
        n_unpredicted = len(df_gt[idx_lb]) - sum(gt_list)
        gt_list += [1] * n_unpredicted
        prob_list += [0] * n_unpredicted
#         print(label)
#         print(gt_list)
#         print(prob_list)
        
        if len(gt_list) > 0:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt_list, prob_list, pos_label=1)
            fpr = fpr[:-1]
            tpr = tpr[:-1]
            fp = fpr * (len(gt_list) - sum(gt_list))
            fps = fp / len(df_predict_lb['seriesuid'].unique())
            frocs[label] = (fps, tpr)
    return frocs


def get_score(frocs):
    scores = {}
    fps_std = np.array([1/8, 1/4, 1/2, 1, 2, 4, 8])
    for label, (fps, tpr) in frocs.items():
        tpr_std = np.interp(fps_std, fps, tpr)
        score = sklearn.metrics.auc(fps_std / fps_std[-1], tpr_std)
        scores[label] = score
    return scores
