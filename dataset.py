import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class DEMDataset(Dataset):

    def __init__(self, file_path, max_samples=None, device=None):
        super().__init__()
        self.file_path = file_path

        try:
            with h5py.File(self.file_path, 'r') as file:
                required_datasets = ['dtm_grp/dst1', 'ori_grp/dst1', 'solar_angles_grp/dst1']
                for dset_name in required_datasets:
                    if dset_name not in file:
                        raise ValueError(f"HDF5 file {self.file_path} is missing dataset {dset_name}")

                num_total_samples = file['dtm_grp']['dst1'].shape[0]

                if (file['ori_grp']['dst1'].shape[0] != num_total_samples or
                        file['solar_angles_grp']['dst1'].shape[0] != num_total_samples):
                    print(f"Warning: Inconsistent lengths for dtm, ori, and solar_angles datasets in file {self.file_path}.")
                    num_total_samples = min(num_total_samples,
                                            file['ori_grp']['dst1'].shape[0],
                                            file['solar_angles_grp']['dst1'].shape[0])

                self.len = min(max_samples, num_total_samples) if max_samples is not None else num_total_samples

        except Exception as e:
            print(f"Error: Could not open or read metadata from HDF5 file {self.file_path}: {e}")
            raise

    def __getitem__(self, index):
        try:
            with h5py.File(self.file_path, 'r') as file:

                def _read_slice_as_float32(dset, idx, tail_shape):
                    try:
                        arr = dset.astype(np.float32)[idx]
                        return np.asarray(arr, dtype=np.float32)
                    except Exception:
                        pass

                    try:
                        out = np.empty(tail_shape, dtype=np.float32)
                        sel = np.s_[idx, ...]
                        dset.read_direct(out, sel)
                        return out
                    except Exception:
                        pass

                    raw = dset[idx]
                    return np.asarray(raw, dtype=np.float32)

                d_dtm = file['dtm_grp']['dst1']
                d_ori = file['ori_grp']['dst1']
                H, W = d_dtm.shape[1], d_dtm.shape[2]

                dtm_data = _read_slice_as_float32(d_dtm, index, (H, W))
                ori_data = _read_slice_as_float32(d_ori, index, (H, W))

                d_ang = file['solar_angles_grp']['dst1']
                solar_angles_data = _read_slice_as_float32(d_ang, index, (2,))

                name = file['filenames'][index]
                name = name.decode('utf-8', errors='ignore') if isinstance(name, (bytes, bytearray)) else str(name)

                dtm = torch.tensor(np.ascontiguousarray(dtm_data, dtype=np.float32), dtype=torch.float32)
                ori = torch.tensor(np.ascontiguousarray(ori_data, dtype=np.float32), dtype=torch.float32)
                solar_angles = torch.tensor(np.ascontiguousarray(solar_angles_data, dtype=np.float32), dtype=torch.float32)

                return dtm, ori, solar_angles, name

        except Exception as e:
            print(f"Error: Could not read data at index {index} from {self.file_path}: {e}")
            raise

    def __len__(self):
        return self.len