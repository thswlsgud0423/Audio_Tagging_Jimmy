# Audio Tagging Project
This project aimas to classify environmental sounds from the ESC-50 dataset using various machinea learning models and feature extraction techniques.

## Dataset
The project uses [ESC-50 dataset](https://github.com/karolpiczak/ESC-50), a labeled collection of 2,000 audio recordings for environmental sound classification.
Key detials include:
- **Total Samples**: 2000 audio clips
- **Audio Length**: 5 sec per clip
- **Classes**: 50 classes organized into 5 main categories

## Feature extraction
For this project, I implemented feature extraction methods:
### Mel Spectrograms: 
- represents the power spectrum of sound, capturing pitch and intensity variations.
**Parameters**: 
**audio_length**: 5 seconds × 32,000 samplesrate = 160000
**n_fft**: The number of FFT (Fast Fourier Transform) components
**hop_length**: The number of samples between successive frames
**win_length**: The size of the window to apply the FFT
**n_mels**: The number of Mel bands to generate
**fmax**: The highest frequency (in Hz) to be considered in the Mel filter bank
    
- **MFCCs (Mel-Frequency Cepstral Coefficients)**: extracts key frequencies that help capture the timbral qualities of audio. (To Be Added if Mel Spectrograms Do Not Work.)

## Preprocessing Steps
1. **Resampling**: audio foles were resamples to a consistent sample rate to standarize processing
2. **Feature Extraction**: transfored into Mel-spectrogram and MFCCs. (+ will possibly add more)
3. **Normalization**: normalized to optimize model performance.

## Models and Results

### CRNN
Input Image -> CNN -> Feature Maps -> Reshape and Permute -> Linear Layer -> LSTM Layer -> Fully Connected Layer -> Class Prediction

| Model Name         | Accuracy                        |
|--------------------|---------------------------------|
| CRNN               | 49.5%  |
| Model 2            |   |
| Model 3            |   |
| Model 4            |   |


# Getting Started
##############################################################
I saw people doing this but also will implement later :>
##############################################################

# Resources

https://github.com/fschmid56/EfficientAT/tree/main

https://github.com/karolpiczak/ESC-50

https://github.com/GitYCC/crnn-pytorch


