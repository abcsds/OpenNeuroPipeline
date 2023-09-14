# OpenNeuro.org EEG pipeline

Given a selected OpenNeuro.org dataset, this pipeline downloads data subject by subject, reads EEG files with MNE, filters between 1 and 40 hz, epochs, and extracts features using MNE-features. 
The parameters for the pipeline are stored in config.json