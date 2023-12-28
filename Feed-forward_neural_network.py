# ============================================================================
# Feed-forward neural network (pytorch)
# Recall fire: 
# False Alarm: 
# ============================================================================
# ----------------------------------------------------------------------------
# fire_filenames
# ----------------------------------------------------------------------------
fire_filenames = [
'sr_1_heptane.csv',
'sr_2_heptane.csv',
'sr_3_heptane.csv',
'sr_4_heptane.csv',
'sr_5_heptane.csv',
'sr_6_heptane.csv',
'sr_7_heptane.csv',
'sr_8_heptane.csv',
'sr_9_heptane.csv',
'sr_8_heptane.csv',
'af9_300m_heptane.csv',
'af9_400m_heptane.csv',
'af9_500m_heptane.csv',
'Fd5Min_1670324284_500m_nheptane_fire.csv',
'Fd5Min_1670325931_800m_nheptane_fire.csv',
'Fd5Min_1670334951_800m_nheptane_fire.csv',
'Fd5Min_1670336466_1000m_nheptane_fire.csv',
'oil_rag_fire_50m.csv',
'oilrag9_50m_dying_fire.csv',
'oilrag8_60m_dying_fire.csv',
'oilrag7_110m.csv',
'oilrag6_80m.csv',
'oilrag5_50m.csv',
'oilrag4_25m.csv',
'oilrag3_7m.csv',
'oilrag2_4m.csv',
'oilrag1_2m.csv',
'small_gas_flame_250m.csv',
'small_gas_flame_70m.csv',
'small_gas_flame_50m.csv',
'oil_rag_fire_100m_1.csv',
'oil_rag_fire_100m_2.csv',
'oil_rag_fire_100m_3.csv',

                  ]
pulse_filenames = [
'filament_bulb_recording_ref.csv',
'halogen_on_off_30s_close.csv',
'halogen_on_off_30s_further.csv',
'halogen_on_off_30s_farthest.csv',
'bulb_2m.csv',
'jw_finger_lr_23_12_06.csv',
'jw_finger_lr_23_12_06_v2.csv',
'jw_finger_lr_23_12_06_v3.csv',
                   ]
welding = [
    'stick_welding_1.csv',
    'stick_welding_2.csv',
]

modulated = [
'26 Sept 2023 - Iron 50cm no steam.csv',
'29 Sept 2023 - iron 2m no steam (no flame detected).csv',
'29 Sept 2023 - iron 1.5m no steam.csv',
'2 Oct 2023 - iron 50cm steam (2).csv',
'2 Oct 2023 - iron 1m steam (2).csv',
'2 Oct 2023 - iron 1.5m steam (2).csv',
'hot_metal_modulated_view.csv',
'intermittent_fan_heater.csv',
'2023_10_18_hafizh_finger_broken_device.csv',
'intermittent_cloudy_sky.csv',
'sun_alarm_shadow_casting.csv',
'Hair dryer fan off heat on 5m.csv',
'Hairdryer fan on 5m.csv',
]
# ----------------------------------------------------------------------------
# Import relevant packages
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import datetime
import os
import numpy as np
import pandas as pd

# Data Preparation
def prepare_data(train_x, train_y, batch_size=16):
    train_x_tensor = torch.FloatTensor(train_x)
    train_y_tensor = torch.LongTensor(train_y)
    dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Model Definition
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training Function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

# Main Execution
if __name__ == "__main__":
    # Load preprocessed data
    data_pickle_path = '/Users/liobaberndt/Desktop/Github/wildfire/preprocessed_data.pkl'
    with open(data_pickle_path, 'rb') as file:
        preprocessed_data = pickle.load(file)

    # Restructure the data
    preprocessed_data = {k: v['LR'] for k, v in preprocessed_data.items()}
    preprocessed_data = {k: [obs_list.reshape(-1) for obs_list in v] for k, v in preprocessed_data.items()}

    # Evaluate classifier using Leave-One-Out (LOO)
    outdir = '/Users/liobaberndt/Desktop/Github/wildfire/classifier_results'
    os.makedirs(outdir, exist_ok=True)
    all_filenames = fire_filenames + pulse_filenames + welding + modulated

    for test_file in all_filenames:
        train_files = [f for f in all_filenames if f != test_file]

        fire_spectra_train = []
        ref_spectra_train = []
        for fname in train_files:
            normed_spectra = preprocessed_data.get(fname, [])
            if fname in fire_filenames:
                fire_spectra_train.extend(normed_spectra)
            else:
                ref_spectra_train.extend(normed_spectra)

        test_normed_spectra = preprocessed_data.get(test_file, [])
        fire_spectra_labels = [1] * len(fire_spectra_train)
        ref_spectra_labels = [0] * len(ref_spectra_train)
        train_x = np.vstack([fire_spectra_train, ref_spectra_train])
        train_y = np.array(fire_spectra_labels + ref_spectra_labels)

        # Prepare data loaders
        train_loader, val_loader = prepare_data(train_x, train_y)

        # Initialize and train the model
        input_dim = train_x.shape[1]
        hidden_dim = 64
        output_dim = 2
        model = FFNN(input_dim, hidden_dim, output_dim)
        train_model(model, train_loader, val_loader)

        # Save or evaluate results as needed
        # ...
