# GEERT: General EEG experimentation in real-time, an open source software for BCI real-time experimental designs
A novel open source software application which covers the main steps
of a BCI paradigm is proposed: real-time acquisition, online EOG arti-
fact removal and band-pass filtering. Being conscious that brain pattern
recognition is under constant development, this software offers the option
to import external python libraries with a predefined structure in order to
include self-developed machine learning approaches, and as experimental
designs require event synchronization, a tcp/ip interface is provided. In
this paper the software is tested on emotion recognition using the Open-
BCI system and a methodology for online EEG emotion estimation. The
application acquired EEG signals in real time from the OpenBCI wearable
system, and with the use of an imported methodology for online EOG ar-
tifact removal, predicted emotional estates. The philosophy behind this
case example is based on a supervised machine learning approach, and
thus offers two modes of interaction: the first allows real-time acquisition
and processing in order to generate a database and build models, while
the second, provides online signal processing using trained models in order
to classify brain patterns. The proposed application proved to be versatile
and easily adaptable to different experimental scenarios while maintain-
ing high performance signal processing in real-time. Wearable devices are
tools of special interest due to the posibilities they offer for BCI related
research such as motor imagery, emotion estimation or attention related
studies, which could benefit from open source applications.

1. Real-time acquisition and visualisation of EEG signals.
2. Trigger synchronisation by a tcp/ip interface which allows start/stop recordings
remotely.
3. Data recording on EDF file format for electrophysiological signals.
4. Online behaviour labelling interface which labels are synchronised and stored on
EDF files.

# INSTRUCTIONS:
1) Install requirements as explained in requirements.txt
2) add permissions: sudo chmod 666 /dev/ttyUSB0 (your serial port)
3) In the folder ./GENERAL/constants_02.py tcp/ip ADDRESS must be set: '10.1.25.82' (your IP)
4) Run BCI_STANDARD_EXPERIMENT_03.py

# CITATION:
@DOI: 10.5281/zenodo.3759306 

# AUTHOR DETAILS AND CONTACT
Author: Mikel Val Calvo
Institution: Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED)
Email: mikel1982mail@gmail.com
