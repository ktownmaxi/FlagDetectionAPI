import os.path

import torch
from PIL import Image
from torchvision import transforms


class FlagPredictionBuilder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", image_size=None, model_path=os.path.join("models", "V3.pth")):
        if image_size is None:
            image_size = [224, 224]
        self.device = device
        self.image_size = image_size
        self.model = torch.load(model_path).to(device)
        self.classnames_file_location = r'C:\Users\ktown\PycharmProjects\FlagDetectionAPI\res\class_names.txt'
        self.classnames = self.read_classnames()
        self.image_transform = self.set_image_transform()

    def read_classnames(self) -> list:
        """
        Reads the class names from the class_names.txt file and returns a list
        :return:
        """
        try:
            items_list = []

            with open(self.classnames_file_location, 'r') as file:
                for line in file:
                    items = line.strip().split(';')
                    items_list.extend(item for item in items if item)

            return items_list

        except FileNotFoundError:
            print("class_names.txt file not found")
            return []

    def set_image_transform(self):
        """
        Creates an image transform and returns it
        :return: Returns the image transform
        """
        image_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return image_transform

    def make_prediction(self, image_path) -> str:
        """
        Classifies an image and returns the predicted class name
        :return: The predicted class name
        """
        img = Image.open(image_path)

        self.model.eval()
        with torch.inference_mode():
            tr_img = self.image_transform(img).unsqueeze(dim=0)
            img_pred = self.model(tr_img.to(self.device))

        target_image_pred_probs = torch.softmax(img_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        label_in_words = self.classnames[target_image_pred_label]

        return label_in_words
