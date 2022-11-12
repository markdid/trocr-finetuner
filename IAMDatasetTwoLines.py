from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch


class IAMDatasetTwoLines(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        # (TrOCRProcesser - from base handwritten pretrained)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        # // 2 is for concatenating item with item+1
        return len(self.df) // 2 - 1

    def concat_v_blank(self, im1, im2, color=(256, 256, 256)):
        dst = Image.new('RGB', (max(im1.width, im2.width),
                        im1.height + im2.height), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    def __getitem__(self, idx):
        # get file name + text
        idx_2 = (idx + 1) * 2
        file_name1 = self.df['file_name'][idx_2 - 1]
        file_name2 = self.df['file_name'][idx_2]
        text = self.df['text'][idx_2 - 1]
        text2 = self.df['text'][idx_2]
        # prepare image (i.e. resize + normalize)
        # image = Image.open(self.root_dir + file_name).convert("RGB")

        image1 = Image.open(self.root_dir + file_name1).convert("RGB")
        image2 = Image.open(self.root_dir + file_name2).convert("RGB")
        image = self.concat_v_blank(image1, image2)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processed = self.processor(image, return_tensors="pt")
        pixel_values = processed.pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text + text2,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label !=
                  self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(),
                    "labels": torch.tensor(labels)}
        return encoding
