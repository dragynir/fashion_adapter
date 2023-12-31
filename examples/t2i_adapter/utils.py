import os
from typing import List
from PIL import Image
from transformers import PretrainedConfig


def image_grid(imgs: List[Image], rows: int, cols: int) -> Image:
    """Creates image grid from list of images."""

    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    # Пример конфига в .cache/huggingface
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
    #     "projection_dim": 768,
    #     "torch_dtype": "float16",
    #     "transformers_version": "4.32.0.dev0",
    #     "vocab_size": 49408
    # }
    # Грузим конфиг текстового энкодера
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- t2iadapter
inference: true
---
    """
    model_card = f"""
# t2iadapter-{repo_id}

These are t2iadapter weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)
