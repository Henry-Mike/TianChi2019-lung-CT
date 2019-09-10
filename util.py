import copy
import math
import os
import pickle

from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.ndimage
import SimpleITK as sitk
from tqdm import tqdm


plt.rcParams['animation.html'] = 'jshtml'

cur_dir = os.path.abspath('.')
root_dir = os.path.dirname(cur_dir)


def plot_ct(image3d, start=0, end=None, ncol=4, annotation_df=None, label_mapper=None):
    """plot ct scan(from SimpleITK) slices as a figure"""

    if end is None:
        end = image3d.shape[0]
    if end <= start: return

    nrow = math.ceil((end - start) / ncol)
    ax_size = 12 / ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ax_size * ncol, ax_size * nrow))
    fig.tight_layout()
    plt.close()

    for i in range(nrow * ncol):
        r, c = divmod(i, ncol)
        ax = axes if nrow == 1 and ncol == 1 else axes[r, c]
        ax.axis('off')

    for i in range(start, end):
        r, c = divmod(i - start, ncol)
        ax = axes if nrow == 1 and ncol == 1 else axes[r, c]
        ax.set_title('slice {}'.format(i))
        ax.imshow(image3d[i], cmap=plt.cm.gray)

        # add annotation
        if annotation_df is not None:
            for row in annotation_df[annotation_df['coordZ'] == i].itertuples():
                x, y, width, height, label = row.coordX, row.coordY, row.diameterX, row.diameterY, row.label
                if label_mapper:
                    label = label_mapper.get(label, label)
                ax.add_patch(plt.Rectangle((x - width/2, y - height/2), width, height, edgecolor='red', facecolor='none'))
                ax.text(x - width/2, y - height/2, label, color='white')

    return fig


def plot_ct_as_slide(image3d, start=0, end=None, interval=300, annotation_df=None, label_mapper=None):
    """plot ct scan(from SimpleITK) as a slide"""

    if end is None:
        end = image3d.shape[0]
    if end <= start: return

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    plt.close()
    ax.axis('off')

    def update(slice_idx):
        ax.clear()
        ax.axis('off')
        ax.text(0.5, 0.5, 'slice {}'.format(slice_idx), color='white', fontsize='x-large', verticalalignment='top')
        ax.imshow(image3d[slice_idx], cmap=plt.cm.gray)

        # add annotation
        if annotation_df is not None:
            for row in annotation_df[annotation_df['coordZ'] == slice_idx].itertuples():
                x, y, width, height, label = row.coordX, row.coordY, row.diameterX, row.diameterY, row.label
                if label_mapper:
                    label = label_mapper.get(label, label)
                ax.add_patch(plt.Rectangle((x - width/2, y - height/2), width, height, edgecolor='red', facecolor='none'))
                ax.text(x - width/2, y - height/2, label, color='white')

    anim = animation.FuncAnimation(fig, update, frames=range(start, end), interval=interval)
    return anim


def normalize(image, ww, wl):
    """normalize ct scan value to 0~1 according to given window-width and window-location"""

    low, high = wl - ww / 2, wl + ww / 2
    image = np.clip(image, low, high)
    image = (image - low) / ww
    return image


def resample(image3d, spacing, new_spacing=[1, 1, 1]):
    """resample ct scan according to spacing of x, y, z axis.
    resample into 1mm x 1mm x 1mm by default.
    return:
        - new image3d
        - resize factor of x, y, z axis
    """

    def rotate_left(lst, n):
        lst = list(lst)
        return lst[n:] + lst[:n]

    resize_factor = np.array(spacing) / new_spacing
    # axis order of SimpleITK image is z-x-y. change resize_factor from x-y-z to z-x-y order.
    resize_factor = np.array(rotate_left(resize_factor, 2))
    new_shape = (image3d.shape * resize_factor).round()
    real_resize_factor = new_shape / image3d.shape
    image3d = scipy.ndimage.interpolation.zoom(image3d, real_resize_factor, mode='nearest')

    real_resize_factor = np.array(rotate_left(real_resize_factor, 1))
    return image3d, real_resize_factor


def resample_annotation(annotation_df, resize_factor):
    """resample annotation dataframe to be consistant with ct scan resampling"""

    annotation_df = annotation_df.copy()
    annotation_df.loc[:, ['coordX', 'coordY', 'coordZ']] = (annotation_df.loc[:, ['coordX', 'coordY', 'coordZ']] * resize_factor).round().astype('int64')
    annotation_df.loc[:, ['diameterX', 'diameterY', 'diameterZ']] = (annotation_df.loc[:, ['diameterX', 'diameterY', 'diameterZ']] * resize_factor).round().astype('int64')
    return annotation_df


def prepare_data_2d(df_record, in_path, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    last_seriesuid = None
    image3d = None
    for row in tqdm(df_record.itertuples()):
        seriesuid, zindex = row.seriesuid, row.coordZ
        if seriesuid != last_seriesuid:
            mhd_file = os.path.join(in_path, '{}.mhd'.format(seriesuid))
            itk_image = sitk.ReadImage(mhd_file)
            image3d = sitk.GetArrayViewFromImage(itk_image)
        image = image3d[zindex]
        np.save(os.path.join(out_path, '{}_{}.npy'.format(seriesuid, zindex)), image)


def prepare_data_3d(df_record, in_path, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    meta = {}
    for seriesuid, df_anno in tqdm(df_record.groupby(['seriesuid'])):
        seriesuid = str(seriesuid)
        mhd_file = os.path.join(in_path, '{}.mhd'.format(seriesuid))
        itk_image = sitk.ReadImage(mhd_file)
        spacing = itk_image.GetSpacing()
        image3d = sitk.GetArrayViewFromImage(itk_image)

        image3d = normalize(image3d, ww=1500, wl=-250)
        image3d, resize_factor = resample(image3d, spacing, [1, 1, 1])
        df_anno = resample_annotation(df_anno, resize_factor)
        meta[seriesuid] = {
            'shape': image3d.shape,
            'spacing': spacing,
            'resize_factor': tuple(resize_factor),
            'boxes': [],
        }

        image_label = np.zeros(image3d.shape, dtype=np.int8)
        for row in df_anno.itertuples():
            z1 = row.coordZ - row.diameterZ // 2
            y1 = row.coordY - row.diameterY // 2
            x1 = row.coordX - row.diameterX // 2
            z2 = z1 + row.diameterZ
            y2 = y1 + row.diameterY
            x2 = x1 + row.diameterX
            image_label[z1:z2, y1:y2, x1:x2] = int(row.label)
            meta[seriesuid]['boxes'].append((z1, y1, x1, z2, y2, x2))

        np.save(os.path.join(out_path, '{}_data.npy'.format(seriesuid)), image3d)
        np.save(os.path.join(out_path, '{}_label.npy'.format(seriesuid)), image_label)

        with open(os.path.join(out_path, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)


def prepare_data_3d_for_test(seriesuid_list, in_path, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    meta = {}
    for seriesuid in tqdm(seriesuid_list):
        seriesuid = str(seriesuid)
        mhd_file = os.path.join(in_path, '{}.mhd'.format(seriesuid))
        itk_image = sitk.ReadImage(mhd_file)
        spacing = itk_image.GetSpacing()
        image3d = sitk.GetArrayViewFromImage(itk_image)

        image3d = normalize(image3d, ww=1500, wl=-250)
        image3d, resize_factor = resample(image3d, spacing, [1, 1, 1])
        meta[seriesuid] = {
            'shape': image3d.shape,
            'spacing': spacing,
            'resize_factor': tuple(resize_factor),
            'boxes': [],
        }
        np.save(os.path.join(out_path, '{}_data.npy'.format(seriesuid)), image3d)

        with open(os.path.join(out_path, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)


# def prepare_data_3d__add_boxes_to_meta(df_record, out_path, old_meta_file='meta.pkl.bak'):
#     with open(os.path.join(out_path, old_meta_file), 'rb') as f:
#         old_meta = pickle.load(f)
#     meta = {}
#     for seriesuid, df_anno in tqdm(df_record.groupby(['seriesuid'])):
#         seriesuid = str(seriesuid)
#         meta[seriesuid] = copy.copy(old_meta[seriesuid])
#         resize_factor = meta[seriesuid]['resize_factor']
#         df_anno = resample_annotation(df_anno, resize_factor)
#         meta[seriesuid]['boxes'] = []

#         for row in df_anno.itertuples():
#             z1 = row.coordZ - row.diameterZ // 2
#             y1 = row.coordY - row.diameterY // 2
#             x1 = row.coordX - row.diameterX // 2
#             z2 = z1 + row.diameterZ
#             y2 = y1 + row.diameterY
#             x2 = x1 + row.diameterX
#             meta[seriesuid]['boxes'].append((z1, y1, x1, z2, y2, x2))

#     with open(os.path.join(out_path, 'meta.pkl'), 'wb') as f:
#         pickle.dump(meta, f)


def parse_prediction_3d(prediction, score_th=0.3, side_len_th=4, labels=(1, 5, 31, 32)):
    targets = []
    for image3d, label in zip(prediction, labels):
        # 每个 image3d 预测一个类别，每个像素值为 0~1，代表该像素属于当前类别的概率
        back = image3d.copy()

        # 首先通过一个阈值进行二值化
        idx = (image3d > score_th)
        image3d[idx] = 1
        image3d[~idx] = 0
        image3d = image3d.astype(np.int8)

        labeled = skimage.measure.label(image3d)
        # 分析连片区域
        blocks = skimage.measure.regionprops(image3d)
        for block in blocks:
            # 筛掉太小的区块
            if block.area < side_len_th ** 3:
                continue

            # 不应该筛掉，应该增大 score_th，使得该区块分解为多个区块
            # # 筛掉孔洞和凹形过多的区块
            # if block.area / block.convex_area < fill_th:
            #     continue

            # 计算得分
            s1, s2, s3, e1, e2, e3 = block.bbox
            box_image = back[s1:e1, s2:e2, s3:e3]
            score = box_image[block.image].mean()

            score *= block.area / block.filled_area
#             score *= block.area / block.convex_area
            score *= block.area / block.bbox_area
            z, y, x = np.round(block.centroid).astype(np.int32)

            targets.append([x, y, z, label, score])
    return targets
