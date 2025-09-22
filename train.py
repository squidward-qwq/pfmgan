import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
from torch.optim import lr_scheduler
import warnings

from generator import GeneratorMiTDecoder, weights_init_normal
from discriminator import Discriminator
from losses import get_loss_functions, SurfaceNormalLoss
from dataset import DEMDataset

import lpips
import piqa


def worker_init_fn(worker_id):
    warnings.filterwarnings("ignore", message=".*On January 1, 2023, MMCV will release v2.0.0.*")


def fix_mit_keys(pretrained_dict):
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        if '.attn.' in k and 'sr' not in k and 'norm' not in k:
            new_k = k.replace('.attn.', '.attn.attn.', 1)
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def format_eta_seconds(seconds):
    if seconds < 0: return "N/A"
    days, rem = divmod(seconds, 86400)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    eta_str = f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"
    if days > 0: eta_str = f"{int(days)}d " + eta_str
    return eta_str


def generate_loss_config_name(loss_weights_dict):
    name_parts = []
    key_map = {
        'GAN': 'gan', 'Pixelwise': 'pix', 'Grad1': 'grad1', 'Grad2': 'grad2',
        'Perceptual': 'lpips', 'WeightedL1': 'wl1', 'SSIM': 'ssim', 'BerHu': 'bh',
        'SurfaceNormal': 'sn', 'TV': 'tv', 'L1': 'l1'
    }
    sorted_keys = sorted(loss_weights_dict.keys())
    for key in sorted_keys:
        weight = loss_weights_dict.get(key, 0)
        if weight > 0:
            short_name = key_map.get(key, key.lower()[:3])
            if isinstance(weight, float):
                weight_str = str(int(weight)) if weight.is_integer() else str(weight).replace('.', 'p')
            else:
                weight_str = str(weight)
            name_parts.append(f"{short_name}{weight_str}")
    if not name_parts: return "no_g_loss"
    return "_".join(name_parts)


if __name__ == '__main__':
    freeze_support()

    EPOCH_START = 0
    N_EPOCHS = 200
    GENERATOR_PRETRAIN_EPOCHS = 20
    H5_FILE_PATH = "./training_dataset.h5"
    MAX_SAMPLES = None
    BATCH_SIZE = 36
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 1e-5
    ADAM_B1 = 0.9
    ADAM_B2 = 0.999
    WEIGHT_DECAY_G = 1e-4
    WEIGHT_DECAY_D = 1e-5
    LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'
    LR_PATIENCE = 10
    LR_FACTOR = 0.5
    N_CPU = 48
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    GENERATOR_IN_CHANNELS = 3
    DEM_OUT_CHANNELS = 1
    SAMPLE_INTERVAL = 1000
    CHECKPOINT_INTERVAL = 10
    OUTPUT_DIR_BASE = "./training_output"
    MIT_MODEL_NAME = 'mit_b4'
    MIT_PRETRAINED_PATH = "./mit_b4.pth"

    if MIT_MODEL_NAME == 'mit_b0':
        DECODER_CHANNELS_LIST = [128, 64, 32, 16]
    else:
        DECODER_CHANNELS_LIST = [256, 128, 64, 32]
    SFF_INTERMEDIATE_CHANNELS = 32

    LOSS_WEIGHTS = {
        'GAN': 1.0, 'Pixelwise': 100.0, 'Grad1': 10.0, 'Grad2': 10.0,
        'Perceptual': 0.0, 'WeightedL1': 0.0, 'SSIM': 0.0, 'BerHu': 0.0,
        'SurfaceNormal': 1.0, 'TV': 0, 'L1': 0.0
    }
    BERHU_DELTA_FACTOR = 0.2
    SURFACE_NORMAL_LOSS_TYPE = 'cosine'
    TV_LOSS_INTERNAL_WEIGHT_FACTOR = 1.0

    loss_config_name_part = generate_loss_config_name(LOSS_WEIGHTS)
    pretrain_tag = f"pre{GENERATOR_PRETRAIN_EPOCHS}" if GENERATOR_PRETRAIN_EPOCHS > 0 else "no_pre"
    DATASET_NAME = f"mars_mit_{MIT_MODEL_NAME}_solar_{pretrain_tag}_{loss_config_name_part}"

    model_save_path = os.path.join(OUTPUT_DIR_BASE, DATASET_NAME)
    os.makedirs(model_save_path, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")

    loss_funcs = get_loss_functions(
        device=device,
        channels=DEM_OUT_CHANNELS,
        berhu_delta_factor=BERHU_DELTA_FACTOR,
        tv_weight_factor=TV_LOSS_INTERNAL_WEIGHT_FACTOR
    )
    if LOSS_WEIGHTS.get('SurfaceNormal', 0) > 0 and loss_funcs.get('SurfaceNormal'):
        loss_funcs['SurfaceNormal'] = SurfaceNormalLoss(
            loss_type=SURFACE_NORMAL_LOSS_TYPE, device=device).to(device)

    generator = GeneratorMiTDecoder(
        in_chans=GENERATOR_IN_CHANNELS,
        out_chans=DEM_OUT_CHANNELS,
        mit_model_name=MIT_MODEL_NAME,
        mit_pretrained_path=MIT_PRETRAINED_PATH,
        decoder_channels=DECODER_CHANNELS_LIST,
        sff_intermediate_channels=SFF_INTERMEDIATE_CHANNELS
    ).to(device)

    if MIT_PRETRAINED_PATH and os.path.exists(MIT_PRETRAINED_PATH):
        try:
            pretrained_weights = torch.load(MIT_PRETRAINED_PATH, map_location=device, weights_only=True)
        except (TypeError, RuntimeError):
            print("Warning: 'weights_only=True' is not supported or failed, trying to load without this parameter.")
            pretrained_weights = torch.load(MIT_PRETRAINED_PATH, map_location=device)

        if 'state_dict' in pretrained_weights:
            pretrained_weights = pretrained_weights['state_dict']

        corrected_weights = fix_mit_keys(pretrained_weights)
        msg = generator.encoder.encoder.load_state_dict(corrected_weights, strict=False)
        print(msg)
    elif MIT_PRETRAINED_PATH:
        print(f"Warning: Pretrained weights path is invalid: {MIT_PRETRAINED_PATH}")

    discriminator = Discriminator(
        dem_channels=DEM_OUT_CHANNELS,
        condition_channels=GENERATOR_IN_CHANNELS
    ).to(device)

    if hasattr(generator, 'decoder') and hasattr(generator.decoder, 'apply'):
        generator.decoder.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=LEARNING_RATE_G, betas=(ADAM_B1, ADAM_B2),
                                    weight_decay=WEIGHT_DECAY_G)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(ADAM_B1, ADAM_B2),
                                    weight_decay=WEIGHT_DECAY_D)

    scheduler_G, scheduler_D = None, None
    if LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
        scheduler_G = lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
        scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

    train_dataset = DEMDataset(file_path=H5_FILE_PATH, max_samples=MAX_SAMPLES)

    init_fn = worker_init_fn if N_CPU > 0 else None

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU,
        pin_memory=cuda,
        drop_last=True,
        worker_init_fn=init_fn
    )

    ten_epoch_accumulator_G_raw = {key: 0.0 for key in LOSS_WEIGHTS.keys()}
    ten_epoch_accumulator_G_weighted_total = 0.0
    ten_epoch_accumulator_D = {'Total': 0.0, 'Real': 0.0, 'Fake': 0.0}
    ten_epoch_batches_processed = 0

    start_time_total_train = time.time()
    for epoch in range(EPOCH_START, N_EPOCHS):
        epoch_start_time = time.time()
        epoch_losses_G_raw = {key: 0.0 for key in LOSS_WEIGHTS.keys()}
        epoch_losses_G_weighted_total = 0.0
        epoch_losses_D = {'Total': 0.0, 'Real': 0.0, 'Fake': 0.0}
        batches_processed_in_epoch = 0

        is_pretrain_phase = epoch < GENERATOR_PRETRAIN_EPOCHS
        if is_pretrain_phase:
            generator.train()
            discriminator.eval()
        else:
            generator.train()
            discriminator.train()

        for i, batch in enumerate(dataloader):
            if batch is None: continue

            target_dem_batch, input_ori_batch, solar_angles_batch, batch_names = batch
            if target_dem_batch is None or input_ori_batch is None or solar_angles_batch is None: continue

            target_dem = target_dem_batch.unsqueeze(1).to(device, non_blocking=True)
            input_ori = input_ori_batch.unsqueeze(1).to(device, non_blocking=True)
            solar_angles = solar_angles_batch.to(device, non_blocking=True)

            B, _, H, W = input_ori.shape
            elevation_channel = solar_angles[:, 0].view(B, 1, 1, 1).expand(B, 1, H, W)
            azimuth_channel = solar_angles[:, 1].view(B, 1, 1, 1).expand(B, 1, H, W)
            generator_input = torch.cat((input_ori, elevation_channel, azimuth_channel), dim=1)

            if i == 0:
                with torch.no_grad():
                    patch_output_shape = discriminator(target_dem, generator_input).shape
                patch_shape = patch_output_shape[1:]

            current_batch_size_actual = generator_input.size(0)
            valid_labels = torch.full((current_batch_size_actual, *patch_shape), 1.0, device=device)
            fake_labels = torch.full((current_batch_size_actual, *patch_shape), 0.0, device=device)

            if not is_pretrain_phase:
                optimizer_D.zero_grad()
                with torch.no_grad():
                    fake_dem_for_D = generator(generator_input)
                pred_real = discriminator(target_dem, generator_input)
                loss_D_real = loss_funcs['GAN'](pred_real, valid_labels)
                pred_fake = discriminator(fake_dem_for_D.detach(), generator_input)
                loss_D_fake = loss_funcs['GAN'](pred_fake, fake_labels)
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
                epoch_losses_D['Total'] += loss_D.item()
                epoch_losses_D['Real'] += loss_D_real.item()
                epoch_losses_D['Fake'] += loss_D_fake.item()

            optimizer_G.zero_grad()
            fake_dem_for_G = generator(generator_input)

            for lf in loss_funcs.values():
                if hasattr(lf, "set_batch_names"):
                    lf.set_batch_names(batch_names)

            current_total_loss_G_weighted = torch.tensor(0.0, device=device)
            current_batch_raw_losses_G = {}

            if not is_pretrain_phase and LOSS_WEIGHTS.get('GAN', 0) > 0:
                pred_fake_adv = discriminator(fake_dem_for_G, generator_input)
                loss_g_gan_raw = loss_funcs['GAN'](pred_fake_adv, valid_labels)
                current_total_loss_G_weighted += loss_g_gan_raw * LOSS_WEIGHTS['GAN']
                epoch_losses_G_raw['GAN'] += loss_g_gan_raw.item()
                current_batch_raw_losses_G['GAN'] = loss_g_gan_raw.item()

            for loss_name, weight in LOSS_WEIGHTS.items():
                if loss_name == 'GAN' or weight <= 0 or not loss_funcs.get(loss_name): continue
                raw_loss_val = loss_funcs[loss_name](fake_dem_for_G, target_dem) if loss_name != 'TV' else loss_funcs[
                    loss_name](fake_dem_for_G)
                current_total_loss_G_weighted += raw_loss_val * weight
                epoch_losses_G_raw[loss_name] += raw_loss_val.item()
                current_batch_raw_losses_G[loss_name] = raw_loss_val.item()

            epoch_losses_G_weighted_total += current_total_loss_G_weighted.item()
            current_total_loss_G_weighted.backward()
            optimizer_G.step()

            batches_processed_in_epoch += 1

            if (i + 1) % SAMPLE_INTERVAL == 0 or (i + 1) == len(dataloader):
                eta_str = format_eta_seconds(
                    ((time.time() - epoch_start_time) / (i + 1)) * (
                            len(dataloader) * (N_EPOCHS - epoch - 1) + (len(dataloader) - i - 1)))

                log_loss_D_str = f"D_Loss: {loss_D.item():.4f}" if not is_pretrain_phase else "D_Loss: N/A"
                g_loss_str = f"G_Total(W): {current_total_loss_G_weighted.item():.4f}"
                g_details_str = ", ".join([f"{key}: {val:.3f}" for key, val in current_batch_raw_losses_G.items()])

                print(
                    f"[Epoch {epoch + 1}/{N_EPOCHS}][Batch {i + 1}/{len(dataloader)}] "
                    f"[{log_loss_D_str}] [{g_loss_str}] "
                    f"| G_Raw: [{g_details_str}] | ETA: {eta_str}"
                )

        if batches_processed_in_epoch > 0:
            print("-" * 80)
            print(f"--- Epoch {epoch + 1}/{N_EPOCHS} Summary (Avg Losses) ---")

            avg_g_weighted_total = epoch_losses_G_weighted_total / batches_processed_in_epoch
            g_raw_avg_str_parts = []
            for name, total_loss in epoch_losses_G_raw.items():
                if LOSS_WEIGHTS.get(name, 0) > 0:
                    avg_raw = total_loss / batches_processed_in_epoch
                    g_raw_avg_str_parts.append(f"{name}: {avg_raw:.4f}")
            print(
                f"  Generator -> Total Weighted: {avg_g_weighted_total:.4f} | Raw Losses: {', '.join(g_raw_avg_str_parts)}")

            if not is_pretrain_phase:
                avg_d_total = epoch_losses_D['Total'] / batches_processed_in_epoch
                avg_d_real = epoch_losses_D['Real'] / batches_processed_in_epoch
                avg_d_fake = epoch_losses_D['Fake'] / batches_processed_in_epoch
                print(f"  Discriminator -> Total: {avg_d_total:.4f} (Real: {avg_d_real:.4f}, Fake: {avg_d_fake:.4f})")

            print("-" * 80)

        if batches_processed_in_epoch > 0:
            ten_epoch_accumulator_G_weighted_total += epoch_losses_G_weighted_total
            for name, total_loss in epoch_losses_G_raw.items():
                ten_epoch_accumulator_G_raw[name] += total_loss
            if not is_pretrain_phase:
                for name, total_loss in epoch_losses_D.items():
                    ten_epoch_accumulator_D[name] += total_loss
            ten_epoch_batches_processed += batches_processed_in_epoch

        if (epoch + 1) % 10 == 0 and epoch > 0:
            if ten_epoch_batches_processed > 0:
                print("\n" + "=" * 80)
                print(f"========== 10-Epoch Avg Summary (Epochs {epoch - 8} to {epoch + 1}) ==========")

                avg_g_10e_weighted = ten_epoch_accumulator_G_weighted_total / ten_epoch_batches_processed
                g_raw_10e_avg_str_parts = []
                for name, total_loss in ten_epoch_accumulator_G_raw.items():
                    if LOSS_WEIGHTS.get(name, 0) > 0:
                        avg_raw_10e = total_loss / ten_epoch_batches_processed
                        g_raw_10e_avg_str_parts.append(f"{name}: {avg_raw_10e:.4f}")
                print(
                    f"  Generator -> Total Weighted: {avg_g_10e_weighted:.4f} | Raw Losses: {', '.join(g_raw_10e_avg_str_parts)}")

                if not is_pretrain_phase or ten_epoch_accumulator_D['Total'] > 0:
                    avg_d_10e_total = ten_epoch_accumulator_D['Total'] / ten_epoch_batches_processed
                    avg_d_10e_real = ten_epoch_accumulator_D['Real'] / ten_epoch_batches_processed
                    avg_d_10e_fake = ten_epoch_accumulator_D['Fake'] / ten_epoch_batches_processed
                    print(
                        f"  Discriminator -> Total: {avg_d_10e_total:.4f} (Real: {avg_d_10e_real:.4f}, Fake: {avg_d_10e_fake:.4f})")

                print("=" * 80 + "\n")

            ten_epoch_accumulator_G_raw = {key: 0.0 for key in LOSS_WEIGHTS.keys()}
            ten_epoch_accumulator_G_weighted_total = 0.0
            ten_epoch_accumulator_D = {'Total': 0.0, 'Real': 0.0, 'Fake': 0.0}
            ten_epoch_batches_processed = 0

        avg_loss_G = epoch_losses_G_weighted_total / batches_processed_in_epoch if batches_processed_in_epoch > 0 else 0
        avg_loss_D = epoch_losses_D[
                         'Total'] / batches_processed_in_epoch if not is_pretrain_phase and batches_processed_in_epoch > 0 else 0

        if scheduler_G and LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
            scheduler_G.step(avg_loss_G)
        elif scheduler_G:
            scheduler_G.step()
        if not is_pretrain_phase:
            if scheduler_D and LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
                scheduler_D.step(avg_loss_D)
            elif scheduler_D:
                scheduler_D.step()

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(generator.state_dict(), f"{model_save_path}/generator_{epoch + 1}.pth")
            if not is_pretrain_phase:
                torch.save(discriminator.state_dict(), f"{model_save_path}/discriminator_{epoch + 1}.pth")
            print(f"Saved checkpoint for Epoch {epoch + 1}")

    torch.save(generator.state_dict(), f"{model_save_path}/generator_final.pth")
    if not any(epoch < GENERATOR_PRETRAIN_EPOCHS for epoch in range(EPOCH_START, N_EPOCHS)):
        torch.save(discriminator.state_dict(), f"{model_save_path}/discriminator_final.pth")
    print(f"Final models saved to {model_save_path}")