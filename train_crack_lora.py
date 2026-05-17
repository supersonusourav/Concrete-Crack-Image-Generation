<<<<<<< HEAD
import argparse
import logging
import math
import os
import glob
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available

logger = get_logger(__name__, log_level="INFO")


class ConcreteCrackDataset(Dataset):
    def __init__(self, data_dir, resolution=512, tokenizer=None):
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.resolution = resolution
        self.tokenizer = tokenizer

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "crack-o-*.jpg")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No .jpg images starting with 'crack-o-' found in {self.images_dir}"
            )

        self.prompt = "a close up photograph of a concrete crack, structural damage"

        self.input_ids = self.tokenizer(
            self.prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename_jpg = os.path.basename(img_path)
        base_name = os.path.splitext(filename_jpg)[0]
        mask_path = os.path.join(self.masks_dir, f"{base_name}.png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing matching mask file. Expected: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return {
            "pixel_values": self.image_transforms(image),
            "mask_values": self.mask_transforms(mask),
            "input_ids": self.input_ids.clone(),
        }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    mask_values = torch.stack([example["mask_values"] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "mask_values": mask_values,
        "input_ids": input_ids,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Concrete Crack LoRA Training Script")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sd-concrete-crack-lora")
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)

    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--mask_loss_weight", type=float, default=3.0)
    parser.add_argument("--use_mask_weighting", action="store_true")

    return parser.parse_args()


def unwrap_model(accelerator, model):
    return accelerator.unwrap_model(model)


def save_lora_weights(accelerator, unet, output_dir):
    unwrapped_unet = unwrap_model(accelerator, unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped_unet)
    )
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("Loading tokenizer, scheduler, and model components...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    logger.info("Adding LoRA adapters to UNet attention layers...")
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory efficient attention enabled.")
            except Exception as e:
                logger.warning(f"Could not enable xFormers: {e}")
        else:
            logger.warning("xFormers is not installed; continuing without it.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled.")

    train_dataset = ConcreteCrackDataset(
        data_dir=args.train_data_dir,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lora_layers = [p for p in unet.parameters() if p.requires_grad]

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using bitsandbytes AdamW8bit optimizer.")
        except ImportError:
            raise ImportError(
                "bitsandbytes is not installed. Install it or remove --use_8bit_adam."
            )
    else:
        optimizer_cls = torch.optim.AdamW
        logger.info("Using standard torch AdamW optimizer.")

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total train batch size = {total_batch_size}")
    logger.info(f"Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new run."
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
        desc="Training Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, 10_000):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device),
                    return_dict=False,
                )[0]

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                target = noise

                if args.use_mask_weighting:
                    loss_map = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    latent_mask = F.interpolate(
                        batch["mask_values"].to(device=accelerator.device, dtype=loss_map.dtype),
                        size=(loss_map.shape[-2], loss_map.shape[-1]),
                        mode="nearest",
                    )
                    weighted_loss = loss_map * (1.0 + args.mask_loss_weight * latent_mask)
                    loss = weighted_loss.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = [p for p in unet.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} old checkpoints."
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint), ignore_errors=True)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        save_lora_weights(accelerator, unet, save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        save_lora_weights(accelerator, unet, args.output_dir)
        logger.info(f"Final LoRA weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
=======
import argparse
import logging
import math
import os
import glob
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available

logger = get_logger(__name__, log_level="INFO")


class ConcreteCrackDataset(Dataset):
    def __init__(self, data_dir, resolution=512, tokenizer=None):
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.resolution = resolution
        self.tokenizer = tokenizer

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "crack-o-*.jpg")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No .jpg images starting with 'crack-o-' found in {self.images_dir}"
            )

        self.prompt = "a close up photograph of a concrete crack, structural damage"

        self.input_ids = self.tokenizer(
            self.prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename_jpg = os.path.basename(img_path)
        base_name = os.path.splitext(filename_jpg)[0]
        mask_path = os.path.join(self.masks_dir, f"{base_name}.png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing matching mask file. Expected: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return {
            "pixel_values": self.image_transforms(image),
            "mask_values": self.mask_transforms(mask),
            "input_ids": self.input_ids.clone(),
        }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    mask_values = torch.stack([example["mask_values"] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "mask_values": mask_values,
        "input_ids": input_ids,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Concrete Crack LoRA Training Script")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sd-concrete-crack-lora")
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)

    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--mask_loss_weight", type=float, default=3.0)
    parser.add_argument("--use_mask_weighting", action="store_true")

    return parser.parse_args()


def unwrap_model(accelerator, model):
    return accelerator.unwrap_model(model)


def save_lora_weights(accelerator, unet, output_dir):
    unwrapped_unet = unwrap_model(accelerator, unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped_unet)
    )
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("Loading tokenizer, scheduler, and model components...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    logger.info("Adding LoRA adapters to UNet attention layers...")
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory efficient attention enabled.")
            except Exception as e:
                logger.warning(f"Could not enable xFormers: {e}")
        else:
            logger.warning("xFormers is not installed; continuing without it.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled.")

    train_dataset = ConcreteCrackDataset(
        data_dir=args.train_data_dir,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lora_layers = [p for p in unet.parameters() if p.requires_grad]

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using bitsandbytes AdamW8bit optimizer.")
        except ImportError:
            raise ImportError(
                "bitsandbytes is not installed. Install it or remove --use_8bit_adam."
            )
    else:
        optimizer_cls = torch.optim.AdamW
        logger.info("Using standard torch AdamW optimizer.")

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total train batch size = {total_batch_size}")
    logger.info(f"Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new run."
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
        desc="Training Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, 10_000):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device),
                    return_dict=False,
                )[0]

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                target = noise

                if args.use_mask_weighting:
                    loss_map = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    latent_mask = F.interpolate(
                        batch["mask_values"].to(device=accelerator.device, dtype=loss_map.dtype),
                        size=(loss_map.shape[-2], loss_map.shape[-1]),
                        mode="nearest",
                    )
                    weighted_loss = loss_map * (1.0 + args.mask_loss_weight * latent_mask)
                    loss = weighted_loss.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = [p for p in unet.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} old checkpoints."
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint), ignore_errors=True)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        save_lora_weights(accelerator, unet, save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        save_lora_weights(accelerator, unet, args.output_dir)
        logger.info(f"Final LoRA weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
>>>>>>> 5571769d2f931f673f57032c6633c1233b71f731
    main()