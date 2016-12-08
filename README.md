# Music Genre Classifier
A Simple Music Genre Classifier

## Welcome
I have recently found the Deep Audio Classification (https://github.com/despoisj/DeepAudioClassification) repository
and i was courious if i can get to work something similar without necessarly use Deep Neural Network to recognize music genres.

### Disclamer
This isn't anything professional, it's just a spare time project i'm developping to learn something about Machine Learning and Data Analysis.
The software is provided as is without any warranty, please read the attached licence.

### Setup
To train the regression you have to place your labeled music into the data/ folder.
At the moment i am writing this i don't have a large enough dataset to provide a "good" model, so you have to do it!
Your file must be named as  *category_progressivenumber.wav*
TO easily convert your .mp3 to .wav install and use sox
```
sox yourfile.mp3 yourfile.wav channels 1
```
If you don't have mp3 support just install *libsox-fmt-mp3* or *libsox-fmt-all*


### Usage
The software is written in Python 3.
I personally recommend to install Continuum Anaconda to have anything ready and working.
If you don't want to install the complete Anaconda package first install dependecies:
```
pip3 install -r requirements.txt
```
Once you've installed it and placed your file in the data folder just run
```
python3 main.py train
```
to train the regression.
To predict your other files add them in te predict folder and run
```
python3 main.py predict
```
At the end your result would be something similar to this:
```
Prediction score (on training set): 0.777777777778
--- Prediction test ---
Classes are
[0] Classical
[50] Other
[100] Metal
- MajorLazerGetFree.wav: [100] - with prob. [[ 0.18406303  0.22849766  0.58743931]]
- PanteraHeresy.wav: [100] - with prob. [[ 0.17705346  0.21830755  0.604639  ]]
- BeethovenOdeToJoy.wav: [50] - with prob. [[ 0.34649982  0.45182985  0.20167033]]
```
Note that the probabily is ordered by the label value.

