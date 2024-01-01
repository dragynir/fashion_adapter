#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers  # Считай новый pytorch timm с множеством мультимодальных моделей (токенайзеры, text-encoders, vae etc.) (https://github.com/huggingface/transformers)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from config import training_config

from utils import import_model_class_from_model_name_or_path, save_model_card


if is_wandb_available():  # проверяем что wandb установлен через importlib.util.find_spec("wandb")
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

# Получаем логгер accelerate, который может работать в multiprocess запуске
logger = get_logger(__name__)


# TODO разобрать во время вызова
def log_validation(vae, unet, adapter, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    adapter = accelerator.unwrap_model(adapter)

    pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        adapter=adapter,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt=validation_prompt, image=validation_image, num_inference_steps=20, generator=generator
                ).images[0]
            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="adapter conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=training_config.pretrained_model_name_or_path,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--adapter_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained adapter model or model identifier from huggingface.co/models."
        " If not specified adapter weights are initialized w.r.t the configurations of SDXL.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=training_config.output_dir,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=training_config.seed, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=training_config.resolution,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--detection_resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=training_config.train_batch_size, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=training_config.max_train_steps,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=training_config.gradient_accumulation_steps,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=training_config.learning_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading."),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=training_config.report_to,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=training_config.mixed_precision,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=training_config.dataset_name,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=training_config.train_data_dir,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the adapter conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=training_config.validation_prompt,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=training_config.validation_image,
        nargs="+",
        help=(
            "A set of paths to the t2iadapter conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=training_config.validation_steps,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_t2iadapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the t2iadapter encoder."
        )

    return args


# TODO
def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


# TODO
# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# TODO
def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[args.image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[args.conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


# TODO
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # Configuration for the Accelerator object based on inner-project needs.
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # Creates an instance of an accelerator for distributed training (on multi-GPU, TPU) or mixed precision training.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # сколько будем накапливать forward пассов для вычисления градиента
        # Смешанная точность — это использование 16-битных и 32-битных типов с плавающей запятой в модели во время обучения, чтобы она работала быстрее и использовала меньше памяти.
        #!!! Подробнее https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
        mixed_precision=args.mixed_precision,  # обучение с ["no", "fp16", "bf16"]
        # Выбираем логгер (wandb или tensorboard)
        log_with=args.report_to,
        # accelerator конфиг
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log Singleton class that has information about the current training environment.
    logger.info(accelerator.state, main_process_only=False)

    # True for one process per server - root process
    if accelerator.is_local_main_process:
        # Настраиваем логирование для главного процесса
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # Настраиваем логирование для остальных
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            # создаем output директорию
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            # Создаем репозиторий на hugging face
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    # Load the tokenizers
    # Загружаем токенайзер из 'stabilityai/stable-diffusion-xl-base-1.0
    #Tokenizers are one of the core components of the NLP pipeline.
    # They serve one purpose: to translate text into data that can be processed by the model.
    # Models can only process numbers, so tokenizers need to convert our text inputs to numerical data.
    # In this section, we’ll explore exactly what happens in the tokenization pipeline.
    # Разбивает текст на токены и кодирует их в индексы токенов
    # Далее индексы можно использовать для индексации эмбеддингов токенов
    #!!! Подробнее про токенайзеры https://huggingface.co/learn/nlp-course/chapter2/4?fw=pt

    # В данном случае используется CLIPTokenizer, он использует  byte-level Byte-Pair-Encoding.
    #!!! Подробнее про Byte-Pair-Encoding (https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)

    # В папке токенайзера лежит vocab.json в котором можно посмотреть
    # Какие слова он закодирует в какие индексы

    # Помимо токенов слов есть специальные токены, например в данном случае
    # bos_token  - начало текста
    # eos_token - конец текста
    # pad_token - токен заполнения (чтобы заполнить контекст)
    # unk_token - если встретили токен которого нет в словаре токенайзера

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",  # папка токенайзера (лежит в .cache/huggingface)
        revision=args.revision,  # The specific model version to use. It can be a branch name, a tag name, or a commit id
        use_fast=False,  # Есть быстрая версия токенайзера
    )
    # SDXL использует два кондишона по тексту; Один и тот же текст кодируется
    # двумя моделями, а резултат конкатенируется перед замешиванием фичей в Unet
    # Поэтому у нас тут два токенайзера
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    # Загружаем классы текстовых энкодеров из их конфигов (лежат в в .cache/huggingface)
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    # This is a fast scheduler which can often generate good outputs in 20-30 steps
    # Нужен для добавления/удаления сгенерированного/предсказанного шума с изображения
    # Далее подробно посмотрим в каких частях пайплайна он участвует
    # Конфиг sheduler из .cache/huggingface

    #!!! Подробнее про реализацию https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51
    #!!! Подробнее видео про сэмплинг https://www.youtube.com/watch?v=HoKDTa5jHvg&t=934s&ab_channel=Outlier
    # {
    #     "_class_name": "EulerDiscreteScheduler",
    #     "_diffusers_version": "0.19.0.dev0",
    #     "beta_end": 0.012,
    #     "beta_schedule": "scaled_linear",
    #     "beta_start": 0.00085,
    #     "clip_sample": false,
    #     "interpolation_type": "linear",
    #     "num_train_timesteps": 1000,
    #     "prediction_type": "epsilon",
    #     "sample_max_value": 1.0,
    #     "set_alpha_to_one": false,
    #     "skip_prk_steps": true,
    #     "steps_offset": 1,
    #     "timestep_spacing": "leading",
    #     "trained_betas": null,
    #     "use_karras_sigmas": false
    # }
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # CLIPTextModel
    # https://arxiv.org/pdf/2103.00020.pdf
    # Модель на основе трансформеров, которая кодирует индексы токенов в эмбеддинги
    # {
    #     "architectures": [
    #         "CLIPTextModel"
    #     ],
    #     "attention_dropout": 0.0,
    #     "bos_token_id": 0,
    #     "dropout": 0.0,
    #     "eos_token_id": 2,
    #     "hidden_act": "quick_gelu",
    #     "hidden_size": 768,
    #     "initializer_factor": 1.0,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 3072,
    #     "layer_norm_eps": 1e-05,
    #     "max_position_embeddings": 77,
    #     "model_type": "clip_text_model",
    #     "num_attention_heads": 12,
    #     "num_hidden_layers": 12,
    #     "pad_token_id": 1,
    #     "projection_dim": 768, # Размер выходного эмбеддинга
    #     "torch_dtype": "float16",
    #     "transformers_version": "4.32.0.dev0",
    #     "vocab_size": 49408
    # }
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, #, variant=args.variant
    )

    # CLIPTextModelWithProjection
    # Модель на основе трансформеров, которая кодирует индексы токенов в эмбеддинги
    # {
    #     "architectures": [
    #         "CLIPTextModelWithProjection"
    #     ],
    #     "attention_dropout": 0.0,
    #     "bos_token_id": 0,
    #     "dropout": 0.0,
    #     "eos_token_id": 2,
    #     "hidden_act": "gelu",
    #     "hidden_size": 1280,
    #     "initializer_factor": 1.0,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 5120,
    #     "layer_norm_eps": 1e-05,
    #     "max_position_embeddings": 77,
    #     "model_type": "clip_text_model",
    #     "num_attention_heads": 20,
    #     "num_hidden_layers": 32,
    #     "pad_token_id": 1,
    #     "projection_dim": 1280,  # Размер выходного эмбеддинга
    #     "torch_dtype": "float16",
    #     "transformers_version": "4.32.0.dev0",
    #     "vocab_size": 49408
    # }
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision #, variant=args.variant
    )



    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    # Variational Autoencoder, который нужен для создания латентов из изображения и наоборот
    # Нужен для сжатия изображения
    #!! Подробнее https://huggingface.co/stabilityai/sdxl-vae
    #!! https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/autoencoders/autoencoder_kl.py#L35
    # Конфиг
    # {
    #     "_class_name": "AutoencoderKL",
    #     "_diffusers_version": "0.20.0.dev0",
    #     "_name_or_path": "../sdxl-vae/",
    #     "act_fn": "silu",
    #     "block_out_channels": [
    #         128,
    #         256,
    #         512,
    #         512
    #     ],
    #     "down_block_types": [
    #         "DownEncoderBlock2D",
    #         "DownEncoderBlock2D",
    #         "DownEncoderBlock2D",
    #         "DownEncoderBlock2D"
    #     ],
    #     "force_upcast": true,
    #     "in_channels": 3,
    #     "latent_channels": 4,
    #     "layers_per_block": 2,
    #     "norm_num_groups": 32,
    #     "out_channels": 3,
    #     "sample_size": 1024,
    #     "scaling_factor": 0.13025,
    #     "up_block_types": [
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D"
    #     ]
    # }
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )

    # Unet like модель для денойзинга на латентах (pretrained)
    # Тут нужно разобраться как добавить свой condition к Unet
    # Что можно использовать для condition
    # down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    # mid_block_additional_residual: Optional[torch.Tensor] = None,
    # down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    # Конфиг
    # {
    #     "_class_name": "UNet2DConditionModel",
    #     "_diffusers_version": "0.19.0.dev0",
    #     "act_fn": "silu",
    #     "addition_embed_type": "text_time",
    #     "addition_embed_type_num_heads": 64,
    #     "addition_time_embed_dim": 256,
    #     "attention_head_dim": [
    #         5,
    #         10,
    #         20
    #     ],
    #     "block_out_channels": [
    #         320,
    #         640,
    #         1280
    #     ],
    #     "center_input_sample": false,
    #     "class_embed_type": null,
    #     "class_embeddings_concat": false,
    #     "conv_in_kernel": 3,
    #     "conv_out_kernel": 3,
    #     "cross_attention_dim": 2048,
    #     "cross_attention_norm": null,
    #     "down_block_types": [
    #         "DownBlock2D",
    #         "CrossAttnDownBlock2D",
    #         "CrossAttnDownBlock2D"
    #     ],
    #     "downsample_padding": 1,
    #     "dual_cross_attention": false,
    #     "encoder_hid_dim": null,
    #     "encoder_hid_dim_type": null,
    #     "flip_sin_to_cos": true,
    #     "freq_shift": 0,
    #     "in_channels": 4,
    #     "layers_per_block": 2,
    #     "mid_block_only_cross_attention": null,
    #     "mid_block_scale_factor": 1,
    #     "mid_block_type": "UNetMidBlock2DCrossAttn",
    #     "norm_eps": 1e-05,
    #     "norm_num_groups": 32,
    #     "num_attention_heads": null,
    #     "num_class_embeds": null,
    #     "only_cross_attention": false,
    #     "out_channels": 4,
    #     "projection_class_embeddings_input_dim": 2816,
    #     "resnet_out_scale_factor": 1.0,
    #     "resnet_skip_time_act": false,
    #     "resnet_time_scale_shift": "default",
    #     "sample_size": 128,
    #     "time_cond_proj_dim": null,
    #     "time_embedding_act_fn": null,
    #     "time_embedding_dim": null,
    #     "time_embedding_type": "positional",
    #     "timestep_post_act": null,
    #     "transformer_layers_per_block": [
    #         1,
    #         2,
    #         10
    #     ],
    #     "up_block_types": [
    #         "CrossAttnUpBlock2D",
    #         "CrossAttnUpBlock2D",
    #         "UpBlock2D"
    #     ],
    #     "upcast_attention": null,
    #     "use_linear_projection": true
    # }

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, #, variant=args.variant
    )

    if args.adapter_model_name_or_path:
        logger.info("Loading existing adapter weights.")
        t2iadapter = T2IAdapter.from_pretrained(args.adapter_model_name_or_path)
    else:
        logger.info("Initializing t2iadapter weights.")

        # Адаптер нового condition
        # Энкодер like модель с resnet блоками
        # Для каждой стадии unet энкодера выдате фичимапу для суммирования
        # Еще есть MultiAdapter, который уже комбинирует conditions
        t2iadapter = T2IAdapter(
            in_channels=3,   # на входе 3-канальное изображение condition
            channels=(320, 640, 1280, 1280),  # каналы в свертах, после которых они пойдут в unet
            num_res_blocks=2,  # количество res блоков в каждом downsample блоке
            downscale_factor=16,  # total downscale входного condition
            adapter_type="full_adapter_xl", # `full_adapter` or `full_adapter_xl` or `light_adapter`.
        )





    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            """Кастомное сохранение весов адаптера."""
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "t2iadapter"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = T2IAdapter.from_pretrained(os.path.join(input_dir, "t2iadapter"))

                if args.control_type != "style":
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)  # Через vae мы не считаем градиенты, поэтому можем его отключить
    text_encoder_one.requires_grad_(False)  # Через текстовые энкодеры мы не считаем градиенты
    text_encoder_two.requires_grad_(False)  # Через текстовые энкодеры мы не считаем градиенты
    t2iadapter.train()  # Будем обучать веса адаптера так, чтобы он влиял на генерацию
    unet.train()  # Оставляем подсчет градиентов т к они нужны будут для адаптера (но в optimizer добавлять unet не будем)


    #!! Подробнее memory efficient attention https://www.photoroom.com/inside-photoroom/stable-diffusion-100-percent-faster-with-memory-efficient-attention
    #!! Подробнее https://arxiv.org/pdf/2205.14135.pdf
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        # Even when we set the batch size to 1 and use gradient accumulation we can still run out of memory when working with large models.
        # In order to compute the gradients during the backward pass all activations from the forward pass are normally saved.
        # This can create a big memory overhead. Alternatively, one could forget all activations during the forward pass and recompute
        # them on demand during the backward pass. This would however add a significant computational overhead and slow down training.
        # Gradient checkpointing strikes a compromise between the two approaches and saves strategically selected activations throughout
        # the computational graph so only a fraction of the activations need to be re-computed for the gradients.
        # See this great article explaining the ideas behind gradient checkpointing.
        #!! Подробнее https://aman.ai/primers/ai/grad-accum-checkpoint/
        #!! Подробнее https://huggingface.co/docs/transformers/v4.18.0/en/performance
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(t2iadapter).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(t2iadapter).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        # Скейлинг learning rate в зависимости от батча (учитываем количество процессов и gradient_accumulation)
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    # В нем есть quantization trick, который не сильно ухудшает качество
    if args.use_8bit_adam:
        # The optimizer is responsible for computing the gradient statistics for back propagation.
        # These calculations are typically done on 32-bit values,
        # but this notebook demonstrates how to use an 8-bit optimizer that saves memory and increases speed
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = t2iadapter.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,  # learning rate
        betas=(args.adam_beta1, args.adam_beta2),  # первый и второй моменты (на сколько доверяем шагу)
        weight_decay=args.adam_weight_decay,  # l2 регуляризация весов
        eps=args.adam_epsilon,  # для стабильности деления
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        # TODO
        original_size = (args.resolution, args.resolution)  # размеры изображения 1024
        target_size = (args.resolution, args.resolution)  # размеры изображения 1024
        # Если захотим обучить чтобы верхний левый угол был не в дефолтном месте, можем поменять
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w) # координаты верхнего левого угла изображения (0, 0) default
        prompt_batch = batch[args.caption_column]

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        # TODO
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma



    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]


    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=args.proportion_empty_prompts,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)  # Hasher that accepts python objects as inputs.
        # Вычисляем заранее эмбеддинги (добавляем в датасет), чтобы не вычислять во время обучения и еще сможем убрать text encoders из памяти
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    t2iadapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        t2iadapter, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(t2iadapter):
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]

                # encode pixel values with batch size of at most 8 to avoid OOM
                latents = []
                for i in range(0, pixel_values.shape[0], 8):
                    latents.append(vae.encode(pixel_values[i : i + 8]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Cubic sampling to sample a random timestep for each image.
                # For more details about why cubic sampling is used, refer to section 3.4 of https://arxiv.org/abs/2302.08453
                timesteps = torch.rand((bsz,), device=latents.device)
                timesteps = (1 - timesteps**3) * noise_scheduler.config.num_train_timesteps
                timesteps = timesteps.long().to(noise_scheduler.timesteps.dtype)
                timesteps = timesteps.clamp(0, noise_scheduler.config.num_train_timesteps - 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Scale the noisy latents for the UNet
                sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Adapter conditioning.
                t2iadapter_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                down_block_additional_residuals = t2iadapter(t2iadapter_image)
                down_block_additional_residuals = [
                    sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
                ]

                # Predict the noise residual
                model_pred = unet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    down_block_additional_residuals=down_block_additional_residuals,
                ).sample

                # Denoise the latents
                denoised_latents = model_pred * (-sigmas) + noisy_latents
                weighing = sigmas**-2.0

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = latents  # we are computing loss against denoise latents
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = t2iadapter.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            unet,
                            t2iadapter,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t2iadapter = accelerator.unwrap_model(t2iadapter)
        t2iadapter.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
