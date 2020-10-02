"""
Mask R-CNN
Train on the ICSI dataset. Code based on the Mask R-CNN example - Balloon.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 icsi.py train --dataset=/path/to/icsi/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 icsi.py train --dataset=/path/to/icsi/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 icsi.py train --dataset=/path/to/icsi/dataset --weights=imagenet

    # Apply color splash to an image
    python3 icsi.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 icsi.py splash --weights=last --video=<URL or path to file>
"""

"""
To see the results of using a trained network on a validation set I used Jupyter notebook files.
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from matplotlib import pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

import cv2
from math import sqrt, pi, fabs

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class ICSIConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "icsi"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + objects

    # Number of training steps per epoch
    # STEPS_PER_EPOCH = 100
    # PG: can be reduced e.g to 5
    STEPS_PER_EPOCH = 2000

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9
    # PG: changed to 0.8
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ICSIDataset(utils.Dataset):

    def load_icsi(self, dataset_dir, subset):
        """Load a subset of the ICSI dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 4 classes to add.
        self.add_class("icsi", 1, "oocyte")
        self.add_class("icsi", 2, "polar body")
        self.add_class("icsi", 3, "spermatozoon")
        self.add_class("icsi", 4, "pipette")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                names = [r['region_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "icsi",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a ICSI dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "icsi":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # PG Assign class_ids by reading class_names    
        class_ids = np.zeros([len(info["polygons"])])
        # In the ICSI dataset, pictures are labeled with name 'komorka', 'cialko', 'plemnik', 'pipeta' representing:
        # oocyte, polar body, spermatozoon, pipette.
        for i, p in enumerate(class_names):
            # "name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'komorka'}
            if p['name'] == 'komorka':
                class_ids[i] = 1
            elif p['name'] == 'cialko':
                class_ids[i] = 2
            elif p['name'] == 'plemnik':
                class_ids[i] = 3
            elif p['name'] == 'pipeta':
                class_ids[i] = 4
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "icsi":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ICSIDataset()
    dataset_train.load_icsi(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ICSIDataset()
    dataset_val.load_icsi(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                # PG: epochs can be reduced e.g. to 3
                epochs=1000,
                layers='heads')


# We don't need splash effect in our implementation because the photos are in grayscale. Code needs refactoring.
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def count_oocyte_area(masks, class_ids, stage, count):
    mask_area = np.sum(masks[:, :, np.where(class_ids == 1)[0]])
    print("Pole dla samego oocytu (class_id = 1): ", mask_area.astype(np.float32))
    f = open("pole_oocytu.txt", "a+")
    f.write("Frame: %d\n" % (count))
    f.write("Pole oocytu, etap {} : %d\r\n".format(stage) % (mask_area.astype(np.float32)))
    f.close()


def count_bbox_coordinates(masks, class_ids, id, label, count):
    bbox_coordinates = utils.extract_bboxes(masks[:, :, np.where(class_ids == id)[0]])
    x1 = bbox_coordinates[0][1]
    x2 = bbox_coordinates[0][3]
    y1 = bbox_coordinates[0][0]
    y2 = bbox_coordinates[0][2]
    #print("{} bbox: ".format(label), bbox_coordinates)
    f = open("bboxes.txt", "a+")
    f.write("Frame: {}".format(count))
    f.write("Bbox {}: {} \r\n".format(label, bbox_coordinates))
    f.close()
    return x1, x2, y1, y2


def count_mask_contours(masks, class_ids, id):
    print("RETR tree: ", cv2.RETR_TREE)
    print("Chain approx simple: ", cv2.CHAIN_APPROX_SIMPLE)
    #print("RETR_FLOODFILL: ", cv2.RETR_FLOODFILL)
    #contours, _ = cv2.findContours(masks[:, :, np.where(class_ids == id)[0]].astype(np.uint8), cv2.RETR_TREE,
    #                               cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(masks[:, :, np.where(class_ids == id)[0]].astype(np.uint8), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    #print("Cnt: ", cnt)
    return cnt


def count_perimeter(cnt):
    perimeter = cv2.arcLength(cnt, True)
    #print("Perimeter: ", perimeter)
    return perimeter


def count_centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    #print("Centroid: ", cx, cy)
    f = open("oocyte_params.txt", "a+")
    f.write("Centroid: ({}, {})\r\n".format(cx, cy))
    f.close()
    return cx, cy


def count_area(cnt):
    area = cv2.contourArea(cnt)
    #print("Area: ", area)
    return area


def count_circularity_ratio(area, perimeter, count):
    circratio = 2 * sqrt(pi * area) / perimeter
    #print("ratio: ", circratio)
    f = open("oocyte_params.txt", "a+")
    f.write("Frame: {}\n".format(count))
    f.write("Circularity ratio: {}\n".format(circratio))
    f.close()
    return circratio



def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    class_names = ['BG', 'oocyte', 'polar body', 'spermatozoon', 'pipette']

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(width, height)
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        print("FPS: ", fps)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        # PG:
        colors = visualize.random_colors(len(class_names))
        while success:
            print("frame: ", count)
            # Read next image
            plt.clf()
            plt.close()
            success, frame = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                frame = frame[..., ::-1]
                # Detect objects
                r = model.detect([frame], verbose=0)[0]
                # Color splash
                splash = color_splash(frame, r['masks'])
                # RGB -> BGR to save image to video
                # PG: splash = splash[..., ::-1]
                frame, labels = visualize.display_instances_video(splash, r['rois'], r['masks'], r['class_ids'],
                                                                  class_names, r['scores'], colors)

                #print("Rois: ", r['rois'])
                #print("Class ids: ", r['class_ids'])

                if 1 in r['class_ids']:
                    x1_oocyte, x2_oocyte, y1_oocyte, y2_oocyte = count_bbox_coordinates(r['masks'], r['class_ids'], 1,
                                                                                        class_names[1], count)

                    cnt = count_mask_contours(r['masks'], r['class_ids'], 1)
                    perimeter = count_perimeter(cnt)
                    area = count_area(cnt)
                    circratio = count_circularity_ratio(area, perimeter, count)
                    cxo, cyo = count_centroid(cnt)


                #if 1 and 2 in r['class_ids']:

                #    cnt_polar = count_mask_contours(r['masks'], r['class_ids'], 2)

                #    cxo, cyo = count_centroid(cnt)
                #    cxp, cyp = count_centroid(cnt_polar)

                #    d = sqrt((cxp - cxo) ** 2 + (cyp - cyo) ** 2)
                #    dx = fabs(cxp - cxo)
                #    dy = cyp - cyo

                if 3 in r['class_ids']:
                    x1, x2, y1, y2 = count_bbox_coordinates(r['masks'], r['class_ids'], 3, class_names[3], count)

                if 4 in r['class_ids']:
                    x1_pipette, x2_pipette, y1_pipette, y2_pipette = count_bbox_coordinates(r['masks'], r['class_ids'],
                                                                                            4, class_names[4], count)
                    #print("Współrzędne końca pipety to x1, y2: (", x1_pipette, ", ", y2_pipette, ")")

                stage_color = (255, 0, 0)

                if labels and 'spermatozoon' in labels and len(set(labels)) == 1 and len(labels) > 1:
                    stage = "Wybor plemnika"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 600), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                elif labels and ('spermatozoon' and 'pipette' in labels) and len(set(labels)) == 2 \
                        and (x1 > x1_pipette) and (y1 > y1_pipette) and (x2 < x2_pipette) and (y2 < y2_pipette):
                    stage = "Pobranie plemnika"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 600), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                elif labels and ('spermatozoon' and 'pipette' in labels) and len(set(labels)) == 2:
                    stage = "Unieruchomienie plemnika"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 600), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                elif labels and ('oocyte' and 'polar body' in labels) and len(set(labels)) == 2:
                    stage = "Ustawienie komorki"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 600), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                elif labels and ('oocyte' and 'pipette' and 'spermatozoon' in labels) and len(set(labels)) >= 3 \
                        and x1_pipette < cxo:
                    stage = "Zaciagniecie zawartosci komorki"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 650), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                elif labels and ('oocyte' and 'pipette' and 'spermatozoon' in labels) and len(set(labels)) >= 3 \
                        and x1 < x1_pipette < x2_oocyte:
                    stage = "Wstrzykniecie plemnika"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 600), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                elif labels and ('oocyte' and 'pipette' in labels) and len(set(labels)) >= 2 \
                        and x1_pipette <= x2_oocyte and circratio < 0.8:
                    stage = "Wbicie pipety"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 600), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                elif labels and ('oocyte' and 'pipette' and 'spermatozoon' in labels) and len(set(labels)) >= 3 \
                        and (x1 > x1_oocyte) and (y1 > y1_oocyte) and (x2 < x2_oocyte) and (y2 < y2_oocyte) and (
                        x1_pipette > x2_oocyte):
                    stage = "Wyciagniecie pipety"
                    frame = cv2.putText(
                        frame, stage, (width - 900, height - 600), cv2.FONT_HERSHEY_COMPLEX, 0.7, stage_color, 2
                    )

                # Add image to video writer
                vwriter.write(frame)
                count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        vcapture.release()
        vwriter.release()
        cv2.destroyAllWindows()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ICSI objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/icsi/dataset/",
                        help='Directory of the ICSI dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ICSIConfig()
    else:
        class InferenceConfig(ICSIConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            # PG zwiekszylam IMAGES_PER_GPU z 1 do 16
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
