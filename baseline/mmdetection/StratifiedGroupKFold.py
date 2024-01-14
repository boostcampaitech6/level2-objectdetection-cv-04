# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import json
import mmengine
import numpy as np
import os

from sklearn.model_selection import StratifiedGroupKFold

prog_description = '''StratifiedGroupKFold trash data split.

To split trash data for StratifiedGroupKFold:
    python StratifiedGroupKFold.py
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of trash dataset.',
        default='/data/ephemeral/home/dataset/')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The output directory for K-fold split annotations.',
        default='/data/ephemeral/home/dataset/k-fold-final/')
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation split.',
        default=10)
    args = parser.parse_args()
    return args

# 데이터셋에서 레이블과 그룹을 추출하기 위한 함수입니다
def extract_labels_and_groups(data):
    labels = []
    groups = []
    for img in data['images']:
        img_annotations = [ann for ann in data['annotations'] if ann['image_id'] == img['id']]
        if img_annotations:
            # 대표 레이블로 첫 번째 주석의 카테고리 ID를 사용합니다
            labels.append(img_annotations[0]['category_id'])
        else:
            labels.append(0)  # 주석이 없는 경우 레이블을 0으로 설정합니다
        groups.append(img['id'])
    return np.array(labels), np.array(groups)

# 주석 파일을 저장하기 위한 함수입니다
def save_anns(name, images, annotations, original_data, out_dir):
    sub_anns = {'images': images, 'annotations': annotations, 'licenses': original_data['licenses'], 'categories': original_data['categories'], 'info': original_data['info']}
    mmengine.mkdir_or_exist(out_dir)
    mmengine.dump(sub_anns, os.path.join(out_dir, name))

# Stratified Group K-Fold를 사용하여 데이터셋을 분할하고 저장하는 함수입니다
def stratified_group_kfold_split(data, out_dir, fold):
    labels, groups = extract_labels_and_groups(data)
    sgkf = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=2024)
    for f, (train_idx, val_idx) in enumerate(sgkf.split(groups, labels, groups), 1):
        train_images = [data['images'][i] for i in train_idx]
        val_images = [data['images'][i] for i in val_idx]
        train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_idx]
        val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_idx]
        save_anns(f'train_fold_{f}.json', train_images, train_annotations, data, out_dir)
        save_anns(f'val_fold_{f}.json', val_images, val_annotations, data, out_dir)


if __name__ == '__main__':
    args = parse_args()
    with open(os.path.join(args.data_root, 'clean-final.json')) as f:
        data = json.load(f)
    stratified_group_kfold_split(data, args.out_dir, args.fold)
