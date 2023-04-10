import lightning.pytorch as pl
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import torchmetrics
import wandb

learning_rate = 1e-2
model_nn = "google/bert_uncased_L-2_H-128_A-2"


class ColaModel(pl.LightningModule):
    def __init__(self, model_name=model_nn, lr=learning_rate):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
        # self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

        self.validation_step_outputs = []

        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(
            num_classes=self.num_classes, task="binary"
        )
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes, task="binary"
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes, task="binary"
        )
        self.precision_micro_metric = torchmetrics.Precision(
            average="micro", task="binary"
        )
        self.recall_micro_metric = torchmetrics.Recall(average="micro", task="binary")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # h_cls = outputs.last_hidden_state[:, 0]
        # logits = self.W(h_cls)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        # self.log("train_loss", loss, prog_bar=True)

        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=labels
        )
        # loss = F.cross_entropy(logits, batch["label"])
        # _, preds = torch.max(logits, dim=1)
        # val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        # val_acc = torch.tensor(val_acc)
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", val_acc, prog_bar=True)

        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        self.log("valid/f1", f1, prog_bar=True)
        self.validation_step_outputs.append(
            {"labels": labels, "logits": outputs.logits}
        )
        return outputs.loss

    # def on_validation_epoch_end(self):
    #     labels = (
    #         (torch.cat([x["labels"] for x in self.validation_step_outputs]))
    #         .detach()
    #         .cpu()
    #         .numpy()
    #     )
    #     logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
    #     preds = torch.argmax(logits, 1).detach().cpu().numpy()

    #     self.logger.experiment.log(
    #         {
    #             "confusion matrix": wandb.plot.confusion_matrix(
    #                 preds=preds, y_true=labels
    #             )
    #         }
    #     )
    #     self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
