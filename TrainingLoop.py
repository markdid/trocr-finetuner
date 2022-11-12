from transformers import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
import torch
import numpy as np
from datasets import load_metric


class TrainingLoop():

    def __init__(self, model, optimizer, device, lr, scheduler, num_epochs=20):

        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs

        self.losses_arr = []
        self.cer_validation_arr = []
        self.validation_arr = []

        self.cer_metric = load_metric("cer")

    def compute_cer(self, pred_ids, label_ids):
        pred_str = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(
            label_ids, skip_special_tokens=True)
        cer = self.cer_metric.compute(
            predictions=pred_str, references=label_str)
        return cer

    def train(self, train_dataloader, eval_dataloader):
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            # train
            train_loss = 0.0
            self.model.train()
            # print("LR: ", self.optimizer)
            with tqdm(train_dataloader) as tepoch:
                for batch in tepoch:

                    for k, v in batch.items():
                        batch[k] = v.to(self.device)

                    outputs = self.model(**batch)
                    logits = outputs.logits

                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_loss += loss.item()

                    loss = outputs.loss
                    self.losses_arr.append(loss)

                    tepoch.set_postfix(
                        loss=loss.item(), child_loss=loss)
                    if epoch < 3:
                        self.scheduler.step()

            print(f"Loss after epoch {epoch}:",
                  train_loss/len(train_dataloader))

            # evaluate

            if epoch % 2 == 0:
                self.model.eval()

                valid_cer = []
                with torch.no_grad():
                    for batch in tqdm(eval_dataloader):
                        # run batch generation
                        outputs = self.model.generate(
                            batch["pixel_values"].to(self.device))
                        # compute metrics
                        cer = self.compute_cer(pred_ids=outputs,
                                               label_ids=batch["labels"])
                        valid_cer.append(cer)

                valid_norm = np.sum(valid_cer) / len(valid_cer)
                print("Validation CER:", valid_norm)
                self.cer_validation_arr.append(valid_norm)

        self.model.save_pretrained(".")
