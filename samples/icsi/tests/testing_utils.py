from math import fabs

from samples.icsi import icsi
from mrcnn import model as modellib
from mrcnn import visualize
from matplotlib import pyplot as plt
import cv2


class InferenceConfig(icsi.ICSIConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


logs = "D:/MASK-RCNN/logs"
weights = "D:/MASK-RCNN/mask_rcnn_icsi_0022.h5"
dataset = icsi.ICSIDataset()
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=logs, config=config)
class_names = ['BG', 'oocyte', 'polar body', 'spermatozoon', 'pipette']


def run_detection(model, dataset, image_id):
    model.load_weights(weights, by_name=True)
    dataset.load_icsi("D:/MASK-RCNN/datasets/icsi", "val")
    dataset.prepare()

    image = dataset.load_image(image_id)
    r = model.detect([image], verbose=0)[0]

    return r, model


def define_stage(r, class_names, count):
    if 1 in r['class_ids'] and len(set(r['class_ids'])) == len(r['class_ids']):
        x1_oocyte, x2_oocyte, y1_oocyte, y2_oocyte = icsi.count_bbox_coordinates(r['masks'], r['class_ids'], 1,
                                                                                 class_names[1])
        cnt = icsi.count_mask_contours(r['masks'], r['class_ids'], 1)
        perimeter = icsi.count_perimeter(cnt)
        area = icsi.count_area(cnt)
        circratio = icsi.count_circularity_ratio(area, perimeter, count)
        cxo, cyo = icsi.count_centroid(cnt, class_names[1])

        if 2 in r['class_ids']:
            x1_polar, x2_polar, y1_polar, y2_polar = icsi.count_bbox_coordinates(r['masks'], r['class_ids'], 2,
                                                                                 class_names[2])
            cnt_polar = icsi.count_mask_contours(r['masks'], r['class_ids'], 2)
            cxp, cyp = icsi.count_centroid(cnt_polar, class_names[2])
            # d = sqrt((cxp - cxo) ** 2 + (cyp - cyo) ** 2)
            dx = fabs(cxp - cxo)
            dy = cyp - cyo
            if dy != 0:
                location = fabs(dx / dy)

    if 3 in r['class_ids']:
        x1, x2, y1, y2 = icsi.count_bbox_coordinates(r['masks'], r['class_ids'], 3, class_names[3])

    if 4 in r['class_ids']:
        x1_pipette, x2_pipette, y1_pipette, y2_pipette = icsi.count_bbox_coordinates(r['masks'], r['class_ids'],
                                                                                     4, class_names[4])

    if r:
        if 3 in r['class_ids'] and len(set(r['class_ids'])) == 1 and len(r['class_ids']) > 1:
            return "Sperm selection"

        elif (3 in r['class_ids']) and (4 in r['class_ids']) and len(set(r['class_ids'])) == 2:
            if (x1 > x1_pipette) and (y1 > y1_pipette) and (x2 < x2_pipette) and (y2 < y2_pipette):
                return "Sperm collection"
            else:
                return "Immobilization of the sperm"

        elif (3 in r['class_ids']) and (1 in r['class_ids']) and (4 in r['class_ids']) and len(set(r['class_ids'])) == len(r['class_ids']):
            if x1_pipette < cxo and (x1 > x1_pipette) and (y1 > y1_pipette) and (x2 < x2_pipette) and (y2 < y2_pipette):
                return "Flow of the cell organelles into the pipette"

            elif x1 < x1_pipette < x2_oocyte and y1 > y1_oocyte:
                return "Sperm injection"

            elif (x1 > x1_oocyte) and (y1 > y1_oocyte) and (x2 < x2_oocyte) and (y2 < y2_oocyte) and (x1_pipette > x2_oocyte):
                return "Removing the pipette"

        elif (2 in r['class_ids']) and len(set(r['class_ids'])) == len(r['class_ids']) == 2:
            if 1 in r['class_ids']:
                if location < 0.5:
                    return "Oocyte positioning"
        elif (4 in r['class_ids']) and (1 in r['class_ids']) and len(set(r['class_ids'])) == len(r['class_ids']):
            if x1_pipette <= x2_oocyte and circratio < 0.85:
                return "Inserting the pipette"
            else:
                return "Stage not detected"
        else:
            return "Stage not detected"


def detect_and_color_splash(model, video_path, stage_name=None):
    model.load_weights(weights, by_name=True)
    if video_path:
        vcapture = cv2.VideoCapture(video_path)

        count = 0
        number_of_ok = 0
        success = True
        colors = visualize.random_colors(len(class_names))

        while success:
            print("frame: ", count)
            plt.clf()
            plt.close()
            success, frame = vcapture.read()

            if success:
                frame = frame[..., ::-1]
                r = model.detect([frame], verbose=0)[0]
                splash = icsi.color_splash(frame, r['masks'])

                frame, labels = visualize.display_instances_video(splash, r['rois'], r['masks'], r['class_ids'],
                                                                  class_names, r['scores'], colors)

                stage = define_stage(r, class_names, count)
                print(stage)

                if stage_name is not None:
                    if stage == stage_name:
                        number_of_ok += 1

                count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        vcapture.release()
        cv2.destroyAllWindows()

    if stage_name is not None:
        print("number of ok: ", number_of_ok)
        correct = number_of_ok / count * 100
        print("{} correct: {}".format(stage_name, correct))

    print("Done")
    return 0


def count_corectness(dataset, class_names):
    stages_ok = ["Sperm injection", "Sperm injection", "Immobilization of the sperm", "Sperm collection",
                 "Immobilization of the sperm", "Immobilization of the sperm", "Sperm selection", "Sperm selection",
                 "Sperm selection", "Stage not detected", "Immobilization of the sperm",
                 "Flow of the cell organelles into the pipette", "Flow of the cell organelles into the pipette",
                 "Flow of the cell organelles into the pipette", "Stage not detected", "Stage not detected",
                 "Sperm injection", "Flow of the cell organelles into the pipette",
                 "Flow of the cell organelles into the pipette", "Flow of the cell organelles into the pipette",
                 "Immobilization of the sperm", "Sperm injection", "Stage not detected", "Removing the pipette",
                 "Removing the pipette", "Removing the pipette", "Stage not detected", "Sperm selection",
                 "Immobilization of the sperm", "Immobilization of the sperm", "Immobilization of the sperm",
                 "Sperm injection", "Immobilization of the sperm", "Stage not detected", "Sperm collection",
                 "Stage not detected", "Stage not detected", "Stage not detected", "Oocyte positioning",
                 "Oocyte positioning", "Oocyte positioning", "Oocyte positioning", "Inserting the pipette",
                 "Inserting the pipette", "Inserting the pipette", "Inserting the pipette", "Inserting the pipette",
                 "Oocyte positioning", "Sperm selection", "Oocyte positioning", "Sperm selection",
                 "Oocyte positioning", "Immobilization of the sperm", "Sperm selection", "Sperm selection",
                 "Sperm collection"
                 ]

    stages_detected = []
    number_of_ok = 0

    model.load_weights(weights, by_name=True)
    dataset.load_icsi("D:/MASK-RCNN/datasets/icsi", "val")
    dataset.prepare()

    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        r = model.detect([image], verbose=0)[0]

        stage_detected = define_stage(r, class_names, 0)
        print("Stage detected for image_id = {}: {}".format(image_id, stage_detected))

        stages_detected.append(stage_detected)

    for s, sd in zip(stages_ok, stages_detected):
        if s == sd:
            number_of_ok += 1
    print("Number of correct stages: ", number_of_ok)

    corectness = number_of_ok / len(stages_ok) * 100
    print("Corectness: ", corectness)

    return corectness
