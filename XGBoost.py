# ============================================================================
# XGboost classifier
# Recall fire: 0.971
# False Alarm: 0.086
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
import pickle
import datetime
import os
import numpy as np
import pandas as pd
import xgboost as xgb  
from sklearn.model_selection import train_test_split
from scipy import stats
# ----------------------------------------------------------------------------
# Evaluate the classifier
# ----------------------------------------------------------------------------
def evaluate_classifier_loo(preprocessed_data):
    outdir = '/Users/liobaberndt/Desktop/Github/wildfire/classifier_results'
    os.makedirs(outdir, exist_ok=True)
    results_rows = []
    all_filenames = fire_filenames + pulse_filenames + welding + modulated
    for test_file in all_filenames:
        train_files = [f for f in all_filenames if f != test_file]
# ----------------------------------------------------------------------------
# Train data preparation
# ----------------------------------------------------------------------------
        fire_spectra_train = []
        ref_spectra_train = []
        for fname in train_files:
            normed_spectra = preprocessed_data.get(fname, [])
            if fname in fire_filenames:
                fire_spectra_train.extend(normed_spectra)
            else:
                ref_spectra_train.extend(normed_spectra)

        test_normed_spectra = preprocessed_data.get(test_file, [])
# ----------------------------------------------------------------------------
# Training the classifier 
# ----------------------------------------------------------------------------
        fire_spectra_labels = [1] * len(fire_spectra_train)
        ref_spectra_labels = [0] * len(ref_spectra_train)
        train_x = np.vstack([fire_spectra_train, ref_spectra_train])
        train_y = np.array(fire_spectra_labels + ref_spectra_labels)
# ----------------------------------------------------------------------------
# XGBoost Classifier Initialization
# ----------------------------------------------------------------------------       
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
        clf.fit(train_x, train_y)
# ----------------------------------------------------------------------------
# Testing the classifier
# ---------------------------------------------------------------------------- 
        if len(test_normed_spectra):
            preds = clf.predict(test_normed_spectra)
            if test_file in fire_filenames:
                print(f'Fire test results for {test_file}: {round(preds.mean(), 3)}')
                target = 1
            else:
                print(f'Ref test results for {test_file}: {round(preds.mean(), 3)}')
                target = 0
            results_rows.append({'experiment': test_file,
                                 'prop_fire': preds.mean(),
                                 'target': target})

    results = pd.DataFrame(results_rows)
    print('recall fire:', round(results[results['target'] == 1].prop_fire.mean() ,3))
    print('false alarm rate:', round(results[results['target'] == 0].prop_fire.mean(), 3))
    fname = f"{outdir}/results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.csv"
    results.to_csv(fname, index=False)

if __name__=="__main__":
    data_pickle_path = '/Users/liobaberndt/Desktop/Github/wildfire/preprocessed_data.pkl'
    with open(data_pickle_path, 'rb') as file:
        preprocessed_data = pickle.load(file)
# ----------------------------------------------------------------------------
# Restructure the data so that it only contains the LR data
# ---------------------------------------------------------------------------- 
    preprocessed_data = {k: v['LR'] for k, v in preprocessed_data.items()}
# ----------------------------------------------------------------------------
# Restructure the data so that it only contains the fire sensor (instead of statistics for all 3):
# ---------------------------------------------------------------------------- 
    preprocessed_data = {k: [obs_list.reshape(-1) for obs_list in v] for k, v in preprocessed_data.items()}
    
    evaluate_classifier_loo(preprocessed_data)
    bp = 1