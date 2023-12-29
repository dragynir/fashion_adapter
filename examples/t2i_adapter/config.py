from dataclasses import dataclass


@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    output_dir: str
    dataset_name: str

    mixed_precision: str
    resolution: int
    learning_rate: float
    max_train_steps: int
    train_batch_size: int
    gradient_accumulation_steps: int

    validation_image: str
    validation_prompt: str
    validation_steps: int
    report_to: str

    seed: int


training_config = TrainingConfig(
    pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0',
    output_dir='./training_logs',
    dataset_name='./fill50k',
    mixed_precision='fp16',
    resolution=1024,
    learning_rate=1e-5,
    max_train_steps=15000,
    train_batch_size=1,
    gradient_accumulation_steps=4,

    validation_image='./validation/conditioning_image_1.png',
    validation_prompt='red circle with blue background',
    validation_steps=100,
    report_to='tensorboard',
    seed=42,
)
