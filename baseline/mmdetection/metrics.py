from map_boxes import mean_average_precision_for_boxes
from pycocotools.coco import COCO
import numpy as np
def meanAveragePrecision(submission, args):
    new_pred = []
    file_names = submission['image_id'].values.tolist()
    bboxes = submission['PredictionString'].values.tolist()
    # check variable type
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f'{file_names[i]} empty box')

    for file_name, bbox in zip(file_names, bboxes):
        boxes = np.array(str(bbox).strip().split(' '))

        # boxes - class ID confidence score xmin ymin xmax ymax
        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        elif isinstance(bbox, float):
            print(f'{file_name} empty box')
            continue
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            # [file_name label_index confidence_score x_min x_max y_min y_max]
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])

    gt = []
    coco = COCO(args.root + args.annotation)
    image_infos = coco.loadImgs(coco.getImgIds())
    for image_info in image_infos:
        annIds = coco.getAnnIds(imgIds = image_info['id'])
        annotation_info_list = coco.loadAnns(annIds)
        for annotation in annotation_info_list:
            """
            기존 bbox annotation
                x_min, y_min, width, height

            -bbox annotation -> gt 재구성-
                class_id, x_min, y_min, x_max, y_max
            """
            # [file_name label_index confidence_score x_min x_max y_min y_max]
            gt.append([image_info['file_name'], 
                    annotation['category_id'], 
                    float(annotation['bbox'][0]), 
                    float(annotation['bbox'][0]+annotation['bbox'][2]), 
                    float(annotation['bbox'][1]), 
                    float(annotation['bbox'][1] + annotation['bbox'][3])])
    return mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)