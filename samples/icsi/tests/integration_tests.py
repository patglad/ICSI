import unittest

import testing_utils


class TestICSI(unittest.TestCase):

    def test_detect_and_color_splash(self):
        video_path = "D:/MASK-RCNN/datasets/videos/103_pobranie_plemnika.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path), 0)

    def test_define_stage(self):
        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 6)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Sperm selection")

        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 2)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Immobilization of the sperm")

        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 55)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Sperm collection")

        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 39)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Oocyte positioning")

        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 46)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Inserting the pipette")

        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 12)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0),
                         "Flow of the cell organelles into the pipette")

        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 1)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Sperm injection")

        r, model = testing_utils.run_detection(testing_utils.model, testing_utils.dataset, 26)
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Stage not detected")

    def test_count_corectness(self):
        corectness = testing_utils.count_corectness(testing_utils.dataset, testing_utils.class_names)
        self.assertIsNotNone(corectness)


if __name__ == '__main__':
    unittest.main()
