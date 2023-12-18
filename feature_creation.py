
"""
This script is used to create the features for the classification task.
It is just intended as an example to give an idea of what the features are
"""
def get_moments_of_signal(signal):
    signal = signal - signal.mean(axis=0)
    # mean = signal.mean()
    # std = signal.std()
    # coeff_var = std / mean
    # Create a histogram
    # hist, bin_edges = np.histogram(signal, bins=30, density=False,range=None)
    # Normalize the histogram to create a probability distribution
    # prob_distribution = hist / hist.sum()
    # entropy = stats.entropy(prob_distribution)
    kurtosis = stats.kurtosis(signal)
    skewness = stats.skew(signal)
    return np.vstack([kurtosis, skewness])

def get_normed_spectra(datafile_content, use_moments=False):
    lr_normed_spectra = []
    sr_normed_spectra = []
    data_section_iterator = DataSectionIterator(datafile_content, settings['fft_size'],
                                                overlap=settings.get('overlap', 0.5))
    for data_segment in iter(data_section_iterator):
        p, s = processor.process_one_time_step(data_segment, sr_subtract=settings.get('sr_subtract', False))
        lr_alarm, sr_alarm = check_alarm_state(settings, s)
        if lr_alarm or sr_alarm:
            ft, frequencies = fft_processing(data_segment, settings['sampling_rate'])
        if lr_alarm:
            # calculate features from Long-Range sensor
            lr_ft = ft[:32, :3]
            # normalise the fft to have unit length
            lr_ft = lr_ft / np.linalg.norm(ft[:,1], axis=0)
            if use_moments:
                moments = get_moments_of_signal(data_segment[:, :3])
                lr_ft = np.concatenate([lr_ft, moments])
            lr_normed_spectra.append(lr_ft)
        if sr_alarm:
            # calculate features from Short-Range sensor
            sr_ft = ft[:32, 3:]
            # normalise the fft to have unit length
            sr_ft = sr_ft / np.linalg.norm(ft[:,4], axis=0)
            if use_moments:
                moments = get_moments_of_signal(data_segment[:, 3:])
                sr_ft = np.concatenate([sr_ft, moments])
            sr_normed_spectra.append(sr_ft)

    return {'LR': lr_normed_spectra, 'SR': sr_normed_spectra}

def preprocess_spectra(filenames, use_moments=False):
    preprocessed_data = {}
    for fname in filenames:
        filepath = get_path_by_name(fname)
        datafile_content = load_data_file(filepath)
        normed_spectra = get_normed_spectra(datafile_content, use_moments=use_moments)
        if len(normed_spectra["LR"]) or len(normed_spectra["SR"]):
            preprocessed_data[fname] = normed_spectra
    return preprocessed_data

# Preprocess the spectra for all files
use_moments=True
data_path = 'out/preprocessed_data.pkl'
if not os.path.exists(data_path):
    preprocessed_data = preprocess_spectra(fire_filenames + pulse_filenames + welding + modulated, use_moments=use_moments)
    # save a pickle of the training data:
    with open(data_path, 'wb') as file:
        pickle.dump(preprocessed_data, file)