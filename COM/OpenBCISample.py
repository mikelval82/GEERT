# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

class OpenBCISample(object):
  def __init__(self, packet_id, channel_data, aux_data):
    """
    Objeto encargado de encapsular cada una de las muestras de la tarjeta OpenBci.
    Encapsula los parámetros de entrada "packet_id", "channel_data" y "aux_data" para
    entregarselos al llamado en "_read_serial_binary" de la clase "OpenBCIBoard".

    """
    self.id = packet_id;
    self.channel_data = channel_data;
    self.aux_data = aux_data;