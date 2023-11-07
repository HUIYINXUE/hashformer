# Fine-tuning a pre-trained RoBERTa model for GLUE
# Based on the following two codes:
#   https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
#   https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb#scrollTo=HzPEIY536tB4
# For comparison:
#   https://github.com/huggingface/transformers/tree/master/examples/text-classification#run-pytorch-version

# Original copyright:
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from datasets import load_from_disk
from typing import Callable, Dict, Optional

import numpy as np
import random
from pathlib import Path



from transformers import AutoConfig, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import transformers
transformers.logging.set_verbosity_debug()

from torch.utils.tensorboard import SummaryWriter
from model import AutoModelForSequenceClassification
from model import EarlyStoppingCallback
from model import (
    DataCollatorForGlue,
    DataCollatorForGlue_Proj,
    DataCollatorForGlue_Conj
)

import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
logger = logging.getLogger(__name__)


from datasets import set_caching_enabled
#set_caching_enabled(False)

# We use dataclass-based configuration objects, let's define the one related to
# which model we are going to train here:
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    emb_mode: str = field(
        default=None,
        metadata={'help': 'The tokenize_mode have to be '
                          + '`pcc`, '
                          + '`de`, '
                          + '`conj`, '
                          + '`ori`. '
                  }
    )
    patience: int = field(
        default=5, metadata={"help": "Patience value for early stopping."}
    )

    #################################################################
    checkpoint_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to a checkpoint."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_pretrained_weights: Optional[bool] = field(
        default=False, metadata={"help": "If `True`, pre-trained weights will not be fine-tuned."}
    )
    model_type: Optional[str] = field(
        default="roberta",
        metadata={"help": "Model type: `roberta` or `bert`."}
    )




class GlueTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = transformers.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            if self.lr_scheduler is None:
                self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=int(num_training_steps * 0.06), num_training_steps=num_training_steps,
                num_cycles=5
                )



def each_tr(seed):
    ##########
    # Configs
    ##########
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(seed)

    fine_tuning_task_dict = {
        'cola': 'cola',
        'mnli': 'mnli',
        'mrpc': 'mrpc',
        'qnli': 'qnli',
        'qqp': 'qqp',
        'rte': 'rte',
        'sst2': 'sst-2',
        'stsb': 'sts-b',
        'wnli': 'wnli'
    }

    report_metrics_dict = {
        'cola': 'eval_mcc',
        'mnli': 'eval_mnli/acc',
        'mrpc': 'eval_acc',
        'qnli': 'eval_acc',
        'qqp': 'eval_f1',
        'rte': 'eval_acc',
        'sst2': 'eval_acc',
        'stsb': 'eval_spearmanr'
    }


    try:
        num_labels = glue_tasks_num_labels[fine_tuning_task_dict[data_args.task_name]]
        output_mode = glue_output_modes[fine_tuning_task_dict[data_args.task_name]]
    except KeyError:
        raise ValueError("Task not found: %s" % (fine_tuning_task_dict[data_args.task_name]))

    ##########
    # Load pretrained model/data and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    ##########
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=fine_tuning_task_dict[data_args.task_name],
        cache_dir=model_args.cache_dir,
    )

    if model_args.emb_mode == 'ori':
        data_collator = DataCollatorForGlue()
    elif model_args.emb_mode == 'pcc':
        data_collator = DataCollatorForGlue_Proj()
    elif model_args.emb_mode == 'de':
        data_collator = DataCollatorForGlue_Proj()
    elif model_args.emb_mode == 'conj':
        data_collator = DataCollatorForGlue_Conj()
    else:
        raise ValueError(
            f"`emb_mode have` to be in `ori`, `pcc`, `de`, `conj`."
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )


    if model_args.freeze_pretrained_weights:
        # freeze pre-trained weights
        for param in getattr(model, model_args.model_type).parameters():
            param.requires_grad = False

    ################################################################# GET DATASETS ###############################################################
    dataset_root_path = Path(data_args.data_dir)
    train_dataset = load_from_disk(dataset_root_path / 'train', keep_in_memory=True)
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = load_from_disk(dataset_root_path / 'dev', keep_in_memory=True)
    test_dataset = load_from_disk(dataset_root_path / 'test', keep_in_memory=True)

    ################################################################# BREAK POINT ##############################################################################################
    ####    dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=data_collator, batch_size=3)
    ####    print(next(iter(dataloader)))
    ####    exit(0)
    ############################################################################################################################################################################
    ##########
    # Set up a evaluation metric
    ##########
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(fine_tuning_task_dict[task_name.split('_')[0]], preds, p.label_ids)
        return compute_metrics_fn

    # Initialize our Trainer
    tb_writer = SummaryWriter(training_args.logging_dir)
    trainer = GlueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tb_writer=tb_writer,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        data_collator=data_collator
    )
    trainer.add_callback(
        EarlyStoppingCallback(
            patience=10000000,
            metric_name=report_metrics_dict[data_args.task_name],
            objective_type="maximize"
        )
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.checkpoint_model_name_or_path
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
#        if trainer.is_world_master():
#            tokenizer.save_pretrained(training_args.output_dir)

    # Show # of paramaters
    if model_args.freeze_pretrained_weights:
        logger.info("Fine-tuned only a linear layer!")
    else:
        logger.info("Fine-tuned all layers!")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total # of trainable params: {total_params}")

    # Evaluation
    eval_results = {}
    logger.info("*** Evaluate ***")
    trainer.compute_metrics = build_compute_metrics_fn(data_args.task_name)
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results_{data_args.task_name}_{training_args.seed}--{seed}.txt"
    )
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(data_args.task_name))
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
    eval_results.update(eval_result)

    # Predict
    predict_results = {}
    logger.info("*** Predict ***")
    trainer.compute_metrics = build_compute_metrics_fn(data_args.task_name)
    predict_result = trainer.evaluate(eval_dataset=test_dataset)

    output_eval_file = os.path.join(
        training_args.output_dir, f"test_results_{data_args.task_name}_{training_args.seed}--{seed}.txt"
    )
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results {} *****".format(data_args.task_name))
            for key, value in predict_result.items():
                logger.info(" %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
    predict_results.update(predict_result)

    output_all_file = os.path.join(
        training_args.output_dir.split('--')[0], f"all_results_{data_args.task_name}_{training_args.seed}.txt"
    )

    return training_args.output_dir.split('--')[0], output_all_file, \
           eval_results[report_metrics_dict[data_args.task_name]], \
           predict_results[report_metrics_dict[data_args.task_name]]




if __name__ == "__main__":
    results_table = []
    for i in range(3):
        output_dir, output_all_file, eval_results, predict_results = each_tr(i)
        results_table.append((eval_results, predict_results))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_all_file, "w") as writer:
        for j in results_table:
            writer.write("%s , %s\n" % j)