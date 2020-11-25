import unittest
import testing_utils


class TestICSIIntegration(unittest.TestCase):

    def test_detect_and_color_splash(self):
        print("Test detect")
        video_path = "D:/MASK-RCNN/datasets/videos/1_pobranie.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path), 0)

        print("Wybor")
        video_path = "D:/MASK-RCNN/datasets/videos/wybor2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Sperm selection"), 0)

        print("Unieruchomienie")
        video_path = "D:/MASK-RCNN/datasets/videos/unieruchomienie2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Immobilization of the sperm"), 0)

        print("Pobranie")
        video_path = "D:/MASK-RCNN/datasets/videos/pobranie2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Sperm collection"), 0)

        print("Ustawienie")
        video_path = "D:/MASK-RCNN/datasets/videos/ustawienie2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Oocyte positioning"), 0)

        print("Wbicie")
        video_path = "D:/MASK-RCNN/datasets/videos/wbicie2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Inserting the pipette"), 0)

        print("Zaciagniecie")
        video_path = "D:/MASK-RCNN/datasets/videos/zaciagniecie2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Flow of the cell organelles into the pipette"), 0)

        print("Wstrzykniecie")
        video_path = "D:/MASK-RCNN/datasets/videos/wstrzykniecie2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Sperm injection"), 0)

        print("Wyciagniecie")
        video_path = "D:/MASK-RCNN/datasets/videos/wyciagniecie2.avi"
        self.assertEqual(testing_utils.detect_and_color_splash(testing_utils.model, video_path, "Removing the pipette"), 0)

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
        self.assertEqual(testing_utils.define_stage(r, testing_utils.class_names, 0), "Stage not detected")

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
