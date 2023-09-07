import argparse

class DetectOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default="data/dataship-jpg", help="path to dataset")
        self.parser.add_argument("--model_name", type=str, default="ship", help="model name")
        self.parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
        self.parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
        self.parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        self.parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        self.parser.add_argument("--number_of_classes", type=int, default=4, help="number of your output classes")
        self.parser.add_argument("--ext", type=str, default="jpg", choices=["png", "jpg"], help="Image file format")

    def parse(self):
        return self.parser.parse_args()
