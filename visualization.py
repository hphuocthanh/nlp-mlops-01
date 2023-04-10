import lightning.pytorch as pl
import torch
import wandb
import pandas as pd


class VisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(
            val_batch["input_ids"].to(device=pl_module.device),
            val_batch["attention_mask"].to(device=pl_module.device),
        )
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {
                "Sentence": sentences,
                "Label": labels.detach().cpu().numpy(),
                "Predicted": preds.detach().cpu().numpy(),
            }
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))

        outputs = pl_module(
            val_batch["input_ids"].to(device=pl_module.device),
            val_batch["attention_mask"].to(device=pl_module.device),
        )
        preds = torch.argmax(outputs.logits, 1).detach().cpu().numpy()
        labels = val_batch["label"].detach().cpu().numpy()

        trainer.logger.experiment.log(
            {
                "confusion matrix": wandb.plot.confusion_matrix(
                    preds=preds, y_true=labels
                )
            }
        )
