import glob
import json
import os
import ast
import cv2
import numpy as np

def deformalize_bbox(bbox, height, width):
    x1, y1, x2, y2 = bbox
    x1 = int(x1 / 1000 * width)
    x2 = int(x2 / 1000 * width)
    y1 = int(y1 / 1000 * height)
    y2 = int(y2 / 1000 * height)
    return [x1, y1, x2, y2]
def get_mask_from_data(data_item, img):
    def convert_bboxes_to_polygons(bboxes, height, width):
        polygons = []
        for bbox in bboxes:
            x1, y1, x2, y2 = deformalize_bbox(bbox, height, width)
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            polygons.append(poly)
        return polygons
    height, width = img.shape[:2]
    bboxes = data_item['bbox']
    polygons = convert_bboxes_to_polygons(bboxes, height, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        cv2.polylines(mask, np.array([poly], dtype=np.int32), True, 1, 1) #在mask上绘制多边形的轮廓
        cv2.fillPoly(mask, np.array([poly], dtype=np.int32), 1) #填充多边形内部
    return mask

def vis_mask(data_item, orig_size):
    mask = get_mask_from_data(data_item, orig_size)
    output_path = "mask_output.png"
    cv2.imwrite(output_path, mask * 255)
    from IPython import embed; embed(); exit()



def draw_black_bboxes_on_image(img_path, data_item):
    """
    Draw black bounding boxes on an image and save the result.

    Args:
    - image_path (str): Path to the input image.
    - bboxes (list): List of bounding boxes, each bbox = [x1, y1, x2, y2].
    - output_path (str): Path to save the output image with drawn boxes.
    """
    # Draw black bounding boxes
    bbox_img = cv2.imread(img_path)
    height, width = img.shape[:2]
    bboxes = data_item['bbox']
    for bbox in bboxes:
        x1, y1, x2, y2 = deformalize_bbox(bbox, height, width)
        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)  # Black color (0, 0, 0)
    return bbox_img



if __name__ == '__main__':
    path = '/cpfs01/user/caixinyu/markdownGenerate/test_bbox/color_block_content_qa.jsonl'
    vis_dir = '/cpfs01/user/chenxiangnan/InternVL_SEG/internvl_chat/vis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    with open(path, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        data_item = json.loads(data[i])
        bbox = data_item['bbox']  
        bbox = ast.literal_eval(bbox)
        data_item['bbox'] = bbox
        img_path = data_item['image']
        img = cv2.imread(img_path)[:, :, ::-1]
        mask = get_mask_from_data(data_item, img)
        ## visualization. Green for target, and red for ignore.
        valid_mask = (mask == 1).astype(np.float32)[:, :, None]
        ignore_mask = (mask == 255).astype(np.float32)[:, :, None]
        vis_img = img * (1 - valid_mask) * (1 - ignore_mask) + (
            (np.array([0, 255, 0]) * 0.6 + img * 0.4) * valid_mask
            + (np.array([255, 0, 0]) * 0.6 + img * 0.4) * ignore_mask
        )
        bbox_img = draw_black_bboxes_on_image(img_path, data_item)
        vis_img = np.concatenate([img, bbox_img, vis_img], 1)
        vis_path = os.path.join(
            vis_dir, os.path.basename(img_path)
        )
        cv2.imwrite(vis_path, vis_img[:, :, ::-1])
        print("Visualization has been saved to: ", vis_path)
        if i > 3:
            break
        
