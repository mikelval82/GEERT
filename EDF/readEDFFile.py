# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""

from __future__ import division, print_function, absolute_import

import os
import numpy as np

import pyedflib


if __name__ == '__main__':
#    f = pyedflib.data.test_generator()
    f = pyedflib.edfreader.EdfReader('./RESULTS/MIKEL/p_trial_0.edf')
    print("\nlibrary version: %s" % pyedflib.version.version)

    print("\ngeneral header:\n")

    # print("filetype: %i\n"%hdr.filetype);
    print("edfsignals: %i" % f.signals_in_file)
    print("file duration: %i seconds" % f.file_duration)
    print("startdate: %i-%i-%i" % (f.getStartdatetime().day,f.getStartdatetime().month,f.getStartdatetime().year))
    print("starttime: %i:%02i:%02i" % (f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second))
    # print("patient: %s" % f.getP);
    # print("recording: %s" % f.getPatientAdditional())
    print("patientcode: %s" % f.getPatientCode())
    print("gender: %s" % f.getGender())
    print("birthdate: %s" % f.getBirthdate())
    print("patient_name: %s" % f.getPatientName())
    print("patient_additional: %s" % f.getPatientAdditional())
    print("admincode: %s" % f.getAdmincode())
    print("technician: %s" % f.getTechnician())
    print("equipment: %s" % f.getEquipment())
    print("recording_additional: %s" % f.getRecordingAdditional())
    print("datarecord duration: %f seconds" % f.getFileDuration())
    print("number of datarecords in the file: %i" % f.datarecords_in_file)
    print("number of annotations in the file: %i" % f.annotations_in_file)
#%%
    channel = 7
    print("\nsignal parameters for the %d.channel:\n\n" % channel)

    print("label: %s" % f.getLabel(channel))
    print("samples in file: %i" % f.getNSamples()[channel])
    # print("samples in datarecord: %i" % f.get
    print("physical maximum: %f" % f.getPhysicalMaximum(channel))
    print("physical minimum: %f" % f.getPhysicalMinimum(channel))
    print("digital maximum: %i" % f.getDigitalMaximum(channel))
    print("digital minimum: %i" % f.getDigitalMinimum(channel))
    print("physical dimension: %s" % f.getPhysicalDimension(channel))
    print("prefilter: %s" % f.getPrefilter(channel))
    print("transducer: %s" % f.getTransducer(channel))
    print("samplefrequency: %f" % f.getSampleFrequency(channel))

    annotations = f.readAnnotations()
    for n in np.arange(f.annotations_in_file):
        print("annotation: onset is %f    duration is %s    description is %s" % (annotations[0][n],annotations[1][n],annotations[2][n]))

    buf = f.readSignal(channel)
   
    import matplotlib.pyplot as plt
    plt.plot(buf)
  
#%%
    
    
#%%    
    f._close()
    del f
