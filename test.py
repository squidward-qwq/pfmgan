import os
import torch
import numpy as np
from osgeo import gdal, osr
import glob
import pandas as pd
from generator import GeneratorMiTDecoder


def min_max_scale(img_array_channel):
    if img_array_channel is None or img_array_channel.size == 0:
        return None
    min_val = np.min(img_array_channel)
    max_val = np.max(img_array_channel)
    range_val = max_val - min_val
    if range_val < 1e-8:
        return np.zeros_like(img_array_channel) if min_val == 0 else (img_array_channel - min_val)
    scaled = (img_array_channel - min_val) / range_val
    return np.clip(scaled, 0.0, 1.0)


def denormalize_dem_to_real_elevation(generated_dem_0_1, target_min_elev, target_max_elev):
    real_elevation_dem = generated_dem_0_1 * (target_max_elev - target_min_elev) + target_min_elev
    return real_elevation_dem


def save_geotiff(output_path, array, geotransform, projection, dtype=gdal.GDT_Float32):
    driver = gdal.GetDriverByName('GTiff')
    if array.ndim == 2:
        rows, cols = array.shape
        num_bands = 1
        array_to_write = array
    elif array.ndim == 3 and array.shape[0] == 1:
        num_bands, rows, cols = array.shape
        array_to_write = array[0]
    else:
        num_bands, rows, cols = array.shape
        array_to_write = array
    dataset = driver.Create(output_path, cols, rows, num_bands, dtype)
    if dataset is None:
        return
    if geotransform:
        dataset.SetGeoTransform(geotransform)
    if projection:
        dataset.SetProjection(projection)
    if num_bands == 1:
        dataset.GetRasterBand(1).WriteArray(array_to_write)
    else:
        for i in range(num_bands):
            dataset.GetRasterBand(i + 1).WriteArray(array_to_write[i])
    dataset.FlushCache()
    dataset = None


def load_min_max_from_csv(csv_path):
    encodings_to_try = ['utf-8-sig', 'gb18030', 'gbk', 'cp936', 'cp932', 'shift_jis', 'big5', 'latin1']
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine='python')
            break
        except Exception:
            continue
    if df is None:
        return None
    df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
    if 'DataType' in df.columns:
        df_labels = df[df['DataType'].astype(str).str.lower() == 'labels'].copy()
        if df_labels.empty:
            df_labels = df.copy()
    else:
        df_labels = df.copy()
    min_max_data = {}
    col_map = {c.lower(): c for c in df_labels.columns}
    fname_col = col_map.get('filename') or col_map.get('file_name') or 'FileName'
    lmin_col = col_map.get('localmin') or 'LocalMin'
    lmax_col = col_map.get('localmax') or 'LocalMax'
    for _, row in df_labels.iterrows():
        if (fname_col not in row) or (lmin_col not in row) or (lmax_col not in row):
            continue
        if pd.isna(row[fname_col]) or pd.isna(row[lmin_col]) or pd.isna(row[lmax_col]):
            continue
        filename_key = str(row[fname_col]).strip()
        try:
            min_max_data[filename_key] = {
                'LocalMin': float(row[lmin_col]),
                'LocalMax': float(row[lmax_col])
            }
        except Exception:
            continue
    return min_max_data


def test_model(generator_weights_path, input_ori_path, output_dir, local_min_max_csv_path,
               normalized_elevation, normalized_azimuth,
               generator_in_channels, dem_out_channels,
               mit_model_name, decoder_channels_list, sff_intermediate_channels):
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")
    local_min_max_data = load_min_max_from_csv(local_min_max_csv_path) if local_min_max_csv_path else None
    generator = GeneratorMiTDecoder(
        in_chans=generator_in_channels, out_chans=dem_out_channels,
        mit_model_name=mit_model_name, mit_pretrained_path=None,
        decoder_channels=decoder_channels_list, sff_intermediate_channels=sff_intermediate_channels
    ).to(device)
    generator.load_state_dict(torch.load(generator_weights_path, map_location=device))
    generator.eval()
    if os.path.isfile(input_ori_path):
        ori_files = [input_ori_path]
    elif os.path.isdir(input_ori_path):
        ori_files = sorted(glob.glob(os.path.join(input_ori_path, '*.tif*')))
    else:
        return
    os.makedirs(output_dir, exist_ok=True)
    for ori_file_path in ori_files:
        ori_basename_with_ext = os.path.basename(ori_file_path)
        ori_basename_no_ext = os.path.splitext(ori_basename_with_ext)[0]
        dtm_filename_key_to_find = ""
        if "-ORI" in ori_basename_no_ext:
            dtm_filename_key_to_find = ori_basename_no_ext.replace("-ORI", "-DTM") + \
                                       os.path.splitext(ori_basename_with_ext)[1]
        target_min_elevation, target_max_elevation = None, None
        if local_min_max_data and dtm_filename_key_to_find in local_min_max_data:
            elev_data = local_min_max_data[dtm_filename_key_to_find]
            target_min_elevation, target_max_elevation = elev_data.get('LocalMin'), elev_data.get('LocalMax')
        try:
            ori_dataset = gdal.Open(ori_file_path)
            original_geotransform, original_projection = ori_dataset.GetGeoTransform(), ori_dataset.GetProjection()
            ori_array = ori_dataset.ReadAsArray().astype(np.float32)
            if ori_array.ndim == 3:
                ori_array = ori_array[0]
            input_tensor_ori = torch.from_numpy(min_max_scale(ori_array)).unsqueeze(0).unsqueeze(0).to(device)
            elevation_channel = torch.full_like(input_tensor_ori, fill_value=float(normalized_elevation))
            azimuth_channel = torch.full_like(input_tensor_ori, fill_value=float(normalized_azimuth))
            generator_input = torch.cat((input_tensor_ori, elevation_channel, azimuth_channel), dim=1)
        except Exception:
            continue
        with torch.no_grad():
            predicted_dem_tensor_0_1 = generator(generator_input)
        dem_output_0_1_np = predicted_dem_tensor_0_1.squeeze(0).squeeze(0).cpu().numpy()
        save_geotiff(os.path.join(output_dir, f"{ori_basename_no_ext}_pred_dem_0_1_relative.tif"), dem_output_0_1_np,
                     original_geotransform, original_projection)
        if target_min_elevation is not None and target_max_elevation is not None and target_max_elevation > target_min_elevation:
            final_real_elevation_dem = denormalize_dem_to_real_elevation(dem_output_0_1_np, target_min_elevation,
                                                                         target_max_elevation)
            save_geotiff(os.path.join(output_dir, f"{ori_basename_no_ext}_pred_dem_real_elevation.tif"),
                         final_real_elevation_dem, original_geotransform, original_projection)


if __name__ == '__main__':
    GENERATOR_WEIGHTS = "./path/to/your/generator_final.pth"
    INPUT_ORI_PATH = "./path/to/your/input_images_or_folder"
    OUTPUT_DEM_DIR = "./predicted_results"
    LOCAL_MIN_MAX_CSV_PATH = "./path/to/your/local_dtm_tile.csv"

    SOLAR_ELEVATION_DEG_TEST = 0
    SOLAR_AZIMUTH_DEG_TEST = 0

    GENERATOR_IN_CHANNELS = 3
    DEM_OUT_CHANNELS = 1
    MIT_MODEL_VARIANT = 'mit_b4'
    DECODER_CHANNELS = [256, 128, 64, 32]
    SFF_INTERMEDIATE_CH = 32

    NORMALIZED_ELEVATION = SOLAR_ELEVATION_DEG_TEST / 90.0
    NORMALIZED_AZIMUTH = SOLAR_AZIMUTH_DEG_TEST / 360.0

    test_model(
        generator_weights_path=GENERATOR_WEIGHTS,
        input_ori_path=INPUT_ORI_PATH,
        output_dir=OUTPUT_DEM_DIR,
        local_min_max_csv_path=LOCAL_MIN_MAX_CSV_PATH,
        normalized_elevation=NORMALIZED_ELEVATION,
        normalized_azimuth=NORMALIZED_AZIMUTH,
        generator_in_channels=GENERATOR_IN_CHANNELS,
        dem_out_channels=DEM_OUT_CHANNELS,
        mit_model_name=MIT_MODEL_VARIANT,
        decoder_channels_list=DECODER_CHANNELS,
        sff_intermediate_channels=SFF_INTERMEDIATE_CH
    )