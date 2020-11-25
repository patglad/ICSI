import unittest

from samples.icsi import icsi
from gui.gui_utils import train_Popen, detection_Popen
import testing_utils


image_id = 43
r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, image_id)
cnt = icsi.count_mask_contours(r['masks'], r['class_ids'], 1)
area = icsi.count_area(cnt)
perimeter = icsi.count_perimeter(cnt)


class TestICSIUnit(unittest.TestCase):

    def test_count_mask_contours(self):
        self.assertIsNotNone(cnt)
        print("Contours: ", cnt)

    def test_count_perimeter(self):
        self.assertIsNotNone(perimeter)
        print("Perimeter: ", perimeter)

    def test_count_centroid(self):
        self.assertIsNotNone(icsi.count_centroid(cnt, "testing"))
        print("Centroid: ", icsi.count_centroid(cnt, "testing"))

    def test_count_area(self):
        self.assertIsNotNone(area)
        print("Area: ", area)

    def test_count_circularity_ratio(self):
        self.assertIsNotNone(icsi.count_circularity_ratio(area, perimeter, "0-testing"))
        print("Circrtio: ", icsi.count_circularity_ratio(area, perimeter, "0-testing"))

    def test_count_bbox_coordinates(self):
        self.assertIsNotNone(icsi.count_bbox_coordinates(r['masks'], r['class_ids'], 2, "polar body"))
        print("Bboxes: ", icsi.count_bbox_coordinates(r['masks'], r['class_ids'], 2, "polar body"))


    def test_train_Popen(self):
        mydataset = "D:/MASK-RCNN/datasets/icsi"
        weights = "D:/MASK-RCNN/mask_rcnn_icsi_0022.h5"
        steps = 1
        epochs = 1
        imGPU = 1
        layers = "heads"
        self.assertEqual(train_Popen(mydataset, weights, epochs, steps, imGPU, layers), 0)

    def test_detection_Popen(self):
        weights = "D:/MASK-RCNN/mask_rcnn_icsi_0022.h5"
        video = "D:/MASK-RCNN/datasets/videos/7_test.avi"
        self.assertEqual(detection_Popen(weights, video), 0)


if __name__ == '__main__':
    unittest.main()
