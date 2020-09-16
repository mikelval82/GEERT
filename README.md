# GEERT: General EEG experimentation in real-time, an open source software for BCI real-time experimental designs
A novel open source software application which covers the main steps
of a BCI paradigm is proposed: real-time acquisition, online EOG arti-
fact removal and band-pass filtering. Being conscious that brain pattern
recognition is under constant development, this software offers the option
to import external python libraries with a predefined structure in order to
include self-developed machine learning approaches, and as experimental
designs require event synchronization, a tcp/ip interface is provided. The philosophy behind this
application is based on a supervised machine learning approach, and
thus offers two modes of interaction: the first allows real-time acquisition
and processing in order to generate a database and build models, while
the second, provides online signal processing using trained models in order
to classify brain patterns. The proposed application is a versatile
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

# DEPENDENCIES:
```
PythonQwt
pyserial
neurodsp
PyQt5
pyqtwebengine
scikit-learn
pandas
ica
scipy
pyqtgraph
pyEDFlib
PyWavelets
lspopt
```

# USE EXAMPLE:
1) add permissions: 
```
sudo chmod 666 /dev/ttyUSB0 (your serial port)
```
2) Run in one terminal:
```
python BCI_STANDARD_EXPERIMENT_03.py
```
3) Set the user filename

4) Set IP and PORT in the app and click the trigger button

5) Run in another terminal:
```
python
```
6) Create a client
```
from COM.trigger_client import trigger_client

tc = trigger_client('IP','PORT')
tc.create_socket()
tc.connect()
```
Then you are ready to start the recording.

```
tc.send_msg(b'start')
```
Labels can be sent asynchronously during the recording and will be stored as events in the EDF user file.

```
tc.send_msg(b'happy')
```

To stop the recording and save the temporal series in the user EDF file.

```
tc.send_msg(b'stop')
```

# CITATION:
@DOI: 10.5281/zenodo.3759306 

# AUTHOR DETAILS AND CONTACT
Author: Mikel Val Calvo
Institution: Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED)
Email: mikel1982mail@gmail.com
