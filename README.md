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


| Model Name         | Accuracy                        |
|--------------------|---------------------------------|
| CRNN               | 49.5%   |
| CNN Model        | 83.75%  |
| CNN Model with epoch = 10       | 89.25%  |
| CNN Model with epoch = 20        | 91.25%  |
| CNN Model with epoch = 50        | 77.75%  |


## Model Details

### CRNN
Input Image -> CNN -> Feature Maps -> Reshape and Permute -> Linear Layer -> LSTM Layer -> Fully Connected Layer -> Class Prediction

### CNN Model
Efficient pre-trained CNN with default settings.

### CNN Model
Efficient pre-trained CNN with custom settings:
    - Audio Length: 160,000
    - n_fft: 1,024
    - Hop Length: 320
    - Window Length: 320
    - n_mels: 128
    - num_epoch: 10 to 50

## Observation
I could clearly see that the model starts to overfitting after 20th epoch - The training accuracy was 100% but the test accuracy went down as we train more.

## How to run

### 1. Clone the Repository  
```bash
git clone https://github.com/thswlsgud0423/Audio_Tagging_Jimmy.git
cd https://github.com/thswlsgud0423/Audio_Tagging_Jimmy.git
```

### 2. Install helpers/models from https://github.com/fschmid56/EfficientAT/tree/main

### 3. Download ESC-50 Dataset and models
Download the dataset and models from https://github.com/karolpiczak/ESC-50 and https://github.com/fschmid56/EfficientAT

### 4. Extract Mel Spectrogram Features
```bash
python mel_spectrogram_extraction
```


### 5. Train and Evaluate the Models
```bash
python run_model.py
```



# Resources

https://github.com/fschmid56/EfficientAT/tree/main

https://github.com/karolpiczak/ESC-50

https://github.com/GitYCC/crnn-pytorch





