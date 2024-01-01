from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    output_dir: str
    dataset_name: str
    train_data_dir: str

    mixed_precision: str
    resolution: int
    learning_rate: float
    max_train_steps: int
    train_batch_size: int
    gradient_accumulation_steps: int

    validation_image: List[str]
    validation_prompt: List[str]
    validation_steps: int
    report_to: str

    seed: int


training_config = TrainingConfig(
    pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0',
    output_dir='./training_logs',
    dataset_name=None,  # './fill50k_hug',  # 'fusing/fill50k',
    train_data_dir='./fill50k_hug',
    mixed_precision='fp16',
    resolution=1024,
    learning_rate=1e-5,
    max_train_steps=15000,
    train_batch_size=1,
    gradient_accumulation_steps=4,

    validation_image=['./validation/conditioning_image_1.png', './conditioning_image_2.png'],
    validation_prompt=['red circle with blue background', 'cyan circle with brown floral background'],
    validation_steps=100,
    report_to='tensorboard',
    seed=42,
)
