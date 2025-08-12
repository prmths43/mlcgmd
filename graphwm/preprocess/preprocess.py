import os
import sys
import time
from pathlib import Path
import multiprocessing as mp
from p_tqdm import p_umap

from battery import load_battery_data
from chain import load_polymer_rg
from graphwm.data.utils import store_data


def polymer_to_h5(data_dir, data_save_dir):
    """
    Process polymer trajectories and save data to .h5 files.
    """
    data_dir = Path(data_dir)
    poly_file_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(poly_file_dirs)} polymer trajectories.")
    print(f"Using {mp.cpu_count()} CPU cores.")
    print("Start processing...")

    def process_one_file(poly_file):
        poly_index = poly_file.name  # safer than parts[-1]
        save_dir = Path(data_save_dir) / poly_index
        save_dir.mkdir(parents=True, exist_ok=True)

        # Skip if output already exists
        if (save_dir / 'bond.h5').exists():
            return

        try:
            data = load_polymer_rg(poly_file)
            store_data(['position'], [data[0]], str(save_dir / 'position.h5'))
            store_data(['lattice'], [data[1]], str(save_dir / 'lattice.h5'))
            store_data(['rgs'], [data[2]], str(save_dir / 'rgs.h5'))
            store_data(['particle_type'], [data[3]], str(save_dir / 'ptype.h5'))
            store_data(['bond_indices'], [data[4]], str(save_dir / 'bond.h5'))
        except Exception as e:
            print(f"Error processing {poly_index}: {e}")

    start_time = time.time()
    # Process the first file to catch issues early
    if poly_file_dirs:
        process_one_file(poly_file_dirs[0])

    # Parallel processing with progress bar
    p_umap(process_one_file, poly_file_dirs)
    elapsed = time.time() - start_time
    print(f"Done. Processed {len(poly_file_dirs)} trajectories in {elapsed:.2f} seconds.")


def battery_to_h5(data_dir, data_save_dir):
    """
    Process battery trajectories and save data to .h5 files.
    """
    data_dir = Path(data_dir)
    poly_file_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(poly_file_dirs)} battery trajectories.")
    print(f"Using {mp.cpu_count()} CPU cores.")
    print("Start processing...")

    def process_one_file(poly_file):
        poly_index = poly_file.name
        save_dir = Path(data_save_dir) / poly_index
        save_dir.mkdir(parents=True, exist_ok=True)

        if (save_dir / 'diffusivity.h5').exists():
            return

        try:
            data = load_battery_data(poly_file)
            store_data(['wrapped_position'], [data[0]], str(save_dir / 'wrapped_position.h5'))
            store_data(['unwrapped_position'], [data[1]], str(save_dir / 'unwrapped_position.h5'))
            store_data(['lattice'], [data[2]], str(save_dir / 'lattice.h5'))
            store_data(['raw_particle_type'], [data[3]], str(save_dir / 'raw_ptype.h5'))
            store_data(['particle_type'], [data[4]], str(save_dir / 'ptype.h5'))
            store_data(['bond_indices'], [data[5]], str(save_dir / 'bond.h5'))
            store_data(['bond_type'], [data[6]], str(save_dir / 'bond_type.h5'))
            store_data(['diffusivity'], [data[7]], str(save_dir / 'diffusivity.h5'))
        except OSError as e:
            print(f"OS error processing {poly_index}: {e}")

    start_time = time.time()
    p_umap(process_one_file, poly_file_dirs)
    elapsed = time.time() - start_time
    print(f"Done. Processed {len(poly_file_dirs)} trajectories in {elapsed:.2f} seconds.")


def protein_to_h5(data_dir, data_save_dir):
    """
    Placeholder for future protein processing.
    """
    pass


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('[Usage] arg1: dataset (chain/battery) arg2: data_dir arg3: save_dir')
        sys.exit(1)
    dataset, data_dir, data_save_dir = sys.argv[1:]
    if dataset == 'chain':
        polymer_to_h5(data_dir, data_save_dir)
    elif dataset == 'battery':
        battery_to_h5(data_dir, data_save_dir)
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)
