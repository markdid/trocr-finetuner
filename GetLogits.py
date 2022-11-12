import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoFeatureExtractor, XLMRobertaTokenizer
from PIL import Image
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from npy_append_array import NpyAppendArray
from tqdm.notebook import tqdm

from IAMDatasetOneLine import IAMDatasetOneLine as IAMDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained(
    'microsoft/trocr-large-handwritten')
model.to(device)
print("Got large model")

df = pd.read_fwf('./IAM/gt_test.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
del df[2]
# some file names end with jp instead of jpg, let's fix this
df['file_name'] = df['file_name'].apply(
    lambda x: x + 'g' if x.endswith('jp') else x)
print(df.head())


def set_model_tokens(model):
    model.config.decoder_start_token_id = 2  # processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.eos_token_id = processor.tokenizer.sep_token_id


set_model_tokens(model)

root_dir = './IAM/'
# df1, df2 = train_test_split(df, test_size=0.5, shuffle=False)

dataset1 = IAMDataset(root_dir=(root_dir + '/image/'),
                      df=df,
                      processor=processor)
dataloader = DataLoader(dataset1, batch_size=1, shuffle=False)

# dataset2 = IAMDataset(root_dir=(root_dir + '/image/'),
#                       df=df2,
#                       processor=processor)

# logits loop
filename1 = 'parent_logits_1.npy'

for batch in tqdm(dataloader):

    # print(i)
    for k, v in batch.items():
        batch[k] = v.to(device)

    out = model(**batch)
    logits = out.logits.cpu().detach().numpy()

    # logit_array.append(logits)
    with NpyAppendArray(filename1) as npaa:
        npaa.append(logits)
