import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# # from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
# from tokenizers.processors import TemplateProcessing


class ExperimentTracker:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        label_encoder,
        args,
        base_dir="experiments",
        metric="f1_score",
        save_interval=50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label_encoder = label_encoder
        self.args = args
        self.base_dir = base_dir
        self.metric = metric
        self.save_interval = save_interval
        self.run_dir = self.create_run_dir()
        self.metrics_tracker = MetricsTracker(
            label_encoder, confusion_interval=save_interval, run_dir=self.run_dir
        )
        self.save_arguments()

    def create_run_dir(self):
        if self.args.experiment_name is None:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = os.path.join(self.base_dir, now)
        else:
            run_dir = os.path.join(self.base_dir, self.args.experiment_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(self.run_dir, "best_model_checkpoint.pt")
        else:
            checkpoint_path = os.path.join(self.run_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            checkpoint_path,
        )

    def save_best_model(self):
        best_model_path = os.path.join(self.run_dir, "best_model.pt")
        torch.save(self.model.state_dict(), best_model_path)

    def update_and_save(
        self,
        epoch,
        train_loss,
        train_preds,
        train_labels,
        test_preds,
        test_labels,
        current_lr,
    ):
        self.metrics_tracker.update(
            train_loss, train_preds, train_labels, test_preds, test_labels, current_lr
        )

        # Save checkpoint every n epochs
        if epoch % self.save_interval == 0:
            self.save_checkpoint(epoch)

        # Check if the current model is the best based on the specified metric
        if (
            self.metrics_tracker.metrics[self.metric][-1]
            == self.metrics_tracker.best_metrics[self.metric]
        ):
            self.save_checkpoint(epoch, is_best=True)

    def save_arguments(self):
        args_dict = vars(self.args)  # Convert the Namespace to a dictionary
        args_path = os.path.join(self.run_dir, "arguments.json")
        with open(args_path, "w") as f:
            json.dump(args_dict, f, indent=4)

    def __str__(self):
        return str(self.metrics_tracker)


class MetricsTracker:
    def __init__(self, label_encoder, confusion_interval=10, run_dir=None):
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "test_acc": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "learning_rate": [],
        }
        self.best_metrics = {
            "train_loss": float("inf"),
            "train_acc": 0,
            "test_acc": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }
        self.label_encoder = label_encoder
        self.confusion_interval = confusion_interval
        self.epoch = 0
        self.run_dir = run_dir

    def update(
        self, train_loss, train_preds, train_labels, test_preds, test_labels, current_lr
    ):
        self.epoch += 1

        # Compute accuracies
        train_acc = accuracy_score(train_labels, train_preds)
        test_acc = accuracy_score(test_labels, test_preds)

        # Update metrics
        self.metrics["train_loss"].append(train_loss)
        self.metrics["train_acc"].append(train_acc)
        self.metrics["test_acc"].append(test_acc)
        self.metrics["learning_rate"].append(current_lr)

        # Compute and update additional metrics
        precision, recall, f1 = self.compute_metrics(test_preds, test_labels)
        self.update_best_metrics(train_loss, train_acc, test_acc, precision, recall, f1)

        if self.epoch % self.confusion_interval == 0:
            self.plot_confusion_matrix(test_preds, test_labels)

        # Save the metrics after each update
        self.save_epoch_metrics()

    def compute_metrics(self, preds, labels, average="binary"):
        precision = precision_score(labels, preds, average=average)
        recall = recall_score(labels, preds, average=average)
        f1 = f1_score(labels, preds, average=average)

        self.metrics["precision"].append(precision)
        self.metrics["recall"].append(recall)
        self.metrics["f1_score"].append(f1)

        return precision, recall, f1

    def update_best_metrics(
        self, train_loss, train_acc, test_acc, precision, recall, f1
    ):
        self.best_metrics["train_loss"] = min(
            self.best_metrics["train_loss"], train_loss
        )
        self.best_metrics["train_acc"] = max(self.best_metrics["train_acc"], train_acc)
        self.best_metrics["test_acc"] = max(self.best_metrics["test_acc"], test_acc)
        self.best_metrics["precision"] = max(self.best_metrics["precision"], precision)
        self.best_metrics["recall"] = max(self.best_metrics["recall"], recall)
        self.best_metrics["f1_score"] = max(self.best_metrics["f1_score"], f1)

    def save_epoch_metrics(self):
        # Save the metrics to a file after each update
        metrics_path = os.path.join(self.run_dir, "epoch_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump({"metrics": self.metrics, "best_metrics": self.best_metrics}, f)

    def plot_confusion_matrix(self, preds, labels):
        string_preds = self.label_encoder.inverse_transform(preds)
        string_labels = self.label_encoder.inverse_transform(labels)
        cm = confusion_matrix(
            string_labels, string_preds, labels=self.label_encoder.classes_
        )
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        if self.run_dir:
            cm_path = os.path.join(self.run_dir, f"cm_epoch_{self.epoch}.pdf")
            plt.savefig(cm_path, format="pdf", bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def __str__(self):
        # Customize the print format
        output = f"Metrics Summary after Epoch {self.epoch}:\n"
        for metric, values in self.metrics.items():
            if metric == "learning_rate":
                output += f"{metric.replace('_', ' ').capitalize()}: {values[-1]:.6f}\n"
            else:
                output += f"{metric.replace('_', ' ').capitalize()}: [{values[-1]:.4f}, {self.best_metrics[metric]:.4f}]\n"
        return output


def set_random_seeds(seed_value=42):
    np.random.seed(seed_value + 1)
    torch.manual_seed(seed_value + 2)
    torch.cuda.manual_seed_all(seed_value + 3)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# def custom_collate_fn(batch):
#     batched_input_ids = []
#     batched_attention_masks = []
#     batched_labels = []
#     batched_chunk_indices = []

# for item in batch:
#     if isinstance(item, list):  # Handle multiple chunks
#         for chunk in item:
#             batched_input_ids.append(chunk["input_ids"])
#             batched_attention_masks.append(chunk["attention_mask"])
#             batched_labels.append(chunk["labels"])
#             batched_chunk_indices.append(chunk["chunk_index"])
#     else:
#         batched_input_ids.append(item["input_ids"])
#         batched_attention_masks.append(item["attention_mask"])
#         batched_labels.append(item["labels"])
#         batched_chunk_indices.append(item["chunk_index"])

# return {
#     "input_ids": torch.stack(batched_input_ids),
#     "attention_mask": torch.stack(batched_attention_masks),
#     "labels": torch.stack(batched_labels),
#     "chunk_index": torch.stack(batched_chunk_indices),
# }


# def build_custom_tokenizer(sequences, mode="numeric"):
#     # Base tokenizer model
#     tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

#     if mode == "numeric":
#         # For numeric syscalls: simple whitespace pre-tokenizer
#         tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#     elif mode == "text":
#         # Custom normalizer to handle common patterns and remove them
#         tokenizer.normalizer = normalizers.Sequence(
#             [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
#         )

#     # Pre-tokenizer to manage different parts of the syscall texts
#     tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
#         [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()]
#     )

# # Common post-processor
# tokenizer.post_processor = TemplateProcessing(
#     single="[CLS] $A [SEP]",
#     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#     special_tokens=[
#         ("[CLS]", 1),
#         ("[SEP]", 2),
#     ],
# )

# # Trainer with special tokens
# trainer = trainers.WordLevelTrainer(
#     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# )

# # Prepare sequences and train the tokenizer
# if mode == "numeric":
#     # Convert numeric sequences to strings
#     str_sequences = [" ".join(map(str, sublist)) for sublist in sequences]
# else:
#     # Join lines into single strings if they are not already
#     str_sequences = [
#         " ".join(sublist) if isinstance(sublist, list) else sublist
#         for sublist in sequences
#     ]

# tokenizer.train_from_iterator(str_sequences, trainer=trainer)
# tokenizer.save("utils/bert-sequences.json")
# return tokenizer


def pool_logits(logits, labels, chunk_indices):
    pooled_logits_dict = {}
    labels_dict = {}

    # Group and pool logits by chunk index
    for idx, (logit, label, chunk_idx) in enumerate(zip(logits, labels, chunk_indices)):
        if chunk_idx.item() not in pooled_logits_dict:
            pooled_logits_dict[chunk_idx.item()] = []
            labels_dict[chunk_idx.item()] = label.item()
        pooled_logits_dict[chunk_idx.item()].append(logit)

    # Average pooling of logits
    pooled_logits = []
    item_labels = []
    for chunk_idx in pooled_logits_dict:
        pooled_logits.append(
            torch.mean(torch.stack(pooled_logits_dict[chunk_idx]), dim=0)
        )
        item_labels.append(labels_dict[chunk_idx])

    # Convert lists to tensors
    pooled_logits = torch.stack(pooled_logits)
    pooled_labels = torch.tensor(item_labels, dtype=torch.long, device=logits.device)

    return pooled_logits, pooled_labels
