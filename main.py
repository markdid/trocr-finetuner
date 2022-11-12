import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW
from PIL import Image
import requests
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from IAMDatasetOneLine import IAMDatasetOneLine as IAMDataset
from IAMDatasetTwoLines import IAMDatasetTwoLines
from TrainingLoop import TrainingLoop


def plot_arr(arr, title="Title"):
    x = np.array(arr)
    plt.title(title)
    plt.plot(x)
    plt.show()


def set_model_tokens(model, processor):
    model.config.decoder_start_token_id = 2  # processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.eos_token_id = processor.tokenizer.sep_token_id


def iam_df():
    df = pd.read_fwf('./IAM/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    del df[2]
    # some file names end with jp instead of jpg, let's fix this
    df['file_name'] = df['file_name'].apply(
        lambda x: x + 'g' if x.endswith('jp') else x)
    return df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = iam_df()
    print(df.head())

    processor = TrOCRProcessor.from_pretrained(
        'microsoft/trocr-small-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained(
        'microsoft/trocr-small-handwritten')
    model.to(device)

    set_model_tokens(model, processor)

    lr = 0.0
    optimizer = AdamW(model.parameters(), lr=(5e-10))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.03)

    root_dir = './IAM/'

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_dataset = IAMDataset(root_dir=(root_dir + '/image/'),
                               df=train_df,
                               processor=processor)
    eval_dataset = IAMDataset(root_dir=(root_dir + '/image/'),
                              df=test_df,
                              processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    trainingloop = TrainingLoop(model=model, optimizer=optimizer,
                                device=device, lr=lr, scheduler=scheduler, num_epochs=20)

    trainingloop.train(train_dataloader, eval_dataloader)

    # plot_arr(trainingloop.losses_arr)
    # plot_arr(trainingloop.cer_validation_arr)


main()
