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
    def __post_init__(self):
        self.pretrained_model_name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.output_dir = './training_logs'
        self.dataset_name = './fill50k'

        self.mixed_precision = 'fp16'
        self.resolution = 1024
        self.learning_rate = 1e-5
        self.max_train_steps = 15000
        self.train_batch_size = 1
        self.gradient_accumulation_steps = 4

        self.validation_image = './validation/conditioning_image_1.png" "./validation/conditioning_image_2.png'
        self.validation_prompt = 'red circle with blue background" "cyan circle with brown floral background'
        self.validation_steps = 100
        self.report_to = 'tensorboard'

        self.seed = 42
