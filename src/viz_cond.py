import os
import openneuro as on
import mne

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Define the dataset ID and download path
dataset_id = "ds004324"
download_path = "./data"

# Re-labeling function: Used to edit the markers in the EEG data
def relabel(events, event_dict):
    e = events.copy()
    reverse_event_dict = {v: k for k,v in event_dict.items()}
    labels = [reverse_event_dict[i] for i in events[:,2]]
    # Remove all events that are not experimental conditions
    idx = [True if "/" in label else False for label in labels]
    e = e[idx,:]
    # Tag all HAPV events as 1 and Rest events as 0
    labels = [reverse_event_dict[i] for i in e[:,2]]
    idx = []
    for label in labels:
        if label.split("/")[0]=="img":
            idx.append(0)
        elif label.split("/")[0]=="sf":
            idx.append(1)
        elif label.split("/")[0]=="toon":
            idx.append(2)
        else:
            raise ValueError(f"Unknown condition {label}")
    l = {"img":0, "sf":1, "toon":2}
    idx = np.array(idx)
    e[:,2] = idx
    return e, l

# Define EEG data processing settings
settings = {
    "rs": 42,
    "l_freq": 0.1, 
    "h_freq": 30,
    "notch_filter": 50,
    "CAR": True,
    "bads": [],
    "outf": "./results/",
    "drop_channels": ['ECG', 'GSR', 'x_dir', 'y_dir', 'z_dir', 'MkIdx'],
    "eog_channels": ["EOGU", "EOGD", "EOGL", "EOGR"],
    "stim_channels": ["MkIdx"],
    "ecg_channels": ["ECG"],
    "misc_channels": ["GSR", "x_dir", "y_dir", "z_dir"],
    "montage": "standard_1020",
    # Window
    "tmin": -0.2,
    "tmax": 0.8,
    "baseline": [-0.2, 0.0],
    "reject_criteria": dict(eeg=500e-6),
    "flat_criteria": dict(eeg=1e-7),
    # Relabeling
    "relabel_func": relabel,
    # Features
    "selected_feats": [
        "mean",  # chnn,
        "variance",  # chnn,
        "std",  # chnn,
        "ptp_amp",  # chnn,
        "skewness",  # chnn,
        "kurtosis",  # chnn,
        "rms",  # chnn,
        "quantile",  # chnn,
        "hurst_exp",  # chnn,
        "app_entropy",  # chnn,
        "samp_entropy",  # chnn,
        "decorr_time",  # chnn,
        "pow_freq_bands",  # band_ch,
        "hjorth_mobility",  # chnn,
        "hjorth_complexity",  # chnn,
        "higuchi_fd",  # chnn,
        "katz_fd",  # chnn,
        "zero_crossings",  # chnn,
        "line_length",  # chnn,
        "spect_entropy",  # chnn,
        "svd_entropy",  # chnn,
        "svd_fisher_info",  # chnn,
        "energy_freq_bands",  # band_ch,
        "spect_edge_freq",  # chnn,
        # "wavelet_coef_energy",  # band_ch, + chnn
        # "teager_kaiser_energy",  # band_ch * 2 + 2,
        # Bivariate
        "max_cross_corr",  # mv_chs_no_self,
        "phase_lock_val",  # mv_chs_no_self,
        # "nonlin_interdep",  # mv_chs_no_self, # Takes too long
        "time_corr",  # mv_chs,
        "spect_corr",  # mv_chs,
        ],
        "subj": {
            'sub-01':{"bads": []},
            'sub-02':{"bads": []},
            'sub-03':{"bads": []},
            'sub-04':{"bads": []},
            'sub-05':{"bads": []},
            'sub-06':{"bads": []},
            'sub-07':{"bads": []},
            'sub-08':{"bads": []},
            'sub-09':{"bads": []},
            'sub-10':{"bads": []},
            'sub-11':{"bads": []},
            'sub-12':{"bads": ["O1", "O2", "Oz"]},
            'sub-13':{"bads": []},
            'sub-14':{"bads": []},
            'sub-15':{"bads": []},
            'sub-16':{"bads": ["CP1"]},
            'sub-17':{"bads": []},
            'sub-18':{"bads": []},
            'sub-19':{"bads": []},
            'sub-20':{"bads": []},
            'sub-21':{"bads": []},
            'sub-22':{"bads": []},
            'sub-23':{"bads": []},
            'sub-24':{"bads": []},
            'sub-25':{"bads": []},
            'sub-26':{"bads": []}
        }
    }

# Create a directory to save results
os.makedirs(settings["outf"], exist_ok=True)

# Download participant data
download_path = "./data"
if not os.path.exists(download_path):
    on.download(dataset=dataset_id, target_dir=download_path, include=["participants.tsv", "participants.json"])

# List all subjects in the dataset
subjects = pd.read_csv(os.path.join(download_path, "participants.tsv"), sep="\t")["participant_id"].values

fig = plt.figure(constrained_layout=True, figsize=(16, 4*len(subjects)))
gs = GridSpec(len(subjects), 3, figure=fig)

# Iterate over subjects and download/process EEG data
for s, subject in enumerate(subjects):
    # subject = subjects[0] # FIXME: Remove
    eeg_file = f"{subject}/ses-01/eeg/{subject}_ses-01_task-RSVP_run-01_eeg.edf"
    eeg_file_path = os.path.join(download_path, eeg_file)
    
    # Download EEG file
    print(f"Downloading EEG file for subject {subject}...")
    if not os.path.exists(eeg_file_path):
        on.download(dataset=dataset_id, target_dir=download_path, include=[f"{subject}/*"])
    
    raw = mne.io.read_raw_edf(eeg_file_path, preload=True)

    if settings["drop_channels"]:
        raw = raw.drop_channels(settings["drop_channels"])

    # # Set EOG reference
    # # FIXME: not working for all subjects
    # mne.set_bipolar_reference(
    #     inst=raw,
    #     anode='EOGD',
    #     cathode='EOGU',
    #     ch_name="HEOG",
    #     drop_refs=True,  # drop anode and cathode from the data
    #     copy=False  # modify in-place
    # )
    # mne.set_bipolar_reference(
    #     inst=raw,
    #     anode='EOGL',
    #     cathode='EOGR',
    #     ch_name="VEOG",
    #     drop_refs=True,  # drop anode and cathode from the data
    #     copy=False  # modify in-place
    # )
    raw.drop_channels(["EOGD", "EOGU", "EOGL", "EOGR"])
    # raw.set_channel_types({"HEOG": "eog", "VEOG": "eog"})
    # eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2), ch_name=["HEOG"], reject_by_annotation=False)
    # eog_epochs.plot_image(combine='mean')
    # eog_epochs.average().plot()
    
    # Set Montage
    if settings["montage"]:
        raw.set_montage(settings["montage"])

    if settings["subj"][subject]["bads"]:
        raw.info["bads"] = settings["subj"][subject]["bads"]
        raw.interpolate_bads()
        # raw.interpolate_bads(, exclude=settings["eog_channels"] + settings["stim_channels"] + settings["ecg_channels"] + settings["misc_channels"])

    ## Filtering
    if settings["notch_filter"]:
        raw.notch_filter(settings["notch_filter"])
    if settings["l_freq"] and settings["h_freq"]:
        raw.filter(l_freq=settings["l_freq"], h_freq=settings["h_freq"])
    elif settings["l_freq"]:
        raw.filter(l_freq=settings["l_freq"])
    elif settings["h_freq"]:
        raw.filter(h_freq=settings["h_freq"])

    ## Rereferencing
    if settings["CAR"]:
        mne.set_eeg_reference(raw, 'average', ch_type="eeg", copy=False)
    
    ## Epoching
    events, event_dict = mne.events_from_annotations(raw)
    assert len(events) == len(set(i["onset"] for i in raw.annotations)), f"Annotations share onset {eeg_file}"
    assert np.abs(np.diff([i["onset"] for i in raw.annotations])).min() > 0.0, f"Annotations share onset {eeg_file}"

    if settings["relabel_func"]:
        events, event_dict = settings["relabel_func"](events, event_dict)

    epochs = mne.Epochs(raw, events,
        event_id=event_dict,
        tmin=settings["tmin"],
        tmax=settings["tmax"],
        baseline=settings["baseline"],
        reject=settings["reject_criteria"],
        flat=settings["flat_criteria"],
        picks="eeg",
        preload=True,
    )


    evk1 = epochs["img"].average()
    evk2 = epochs["sf"].average()
    evk3 = epochs["toon"].average()

    # Plot both conditions as subfigures of a larger mpl figure

    subfigure1 = fig.add_subplot(gs[s, 0])
    subfigure1.set_title("IMG")
    evk1.plot(axes=subfigure1, show=False)

    subfigure2 = fig.add_subplot(gs[s, 1])
    subfigure2.set_title("SF")
    evk2.plot(axes=subfigure2, show=False)

    subfigure3 = fig.add_subplot(gs[s, 2])
    subfigure3.set_title("TOON")
    evk3.plot(axes=subfigure3, show=False)


plt.savefig(os.path.join(settings["outf"], "img", "cond_all.png"), dpi=300)
