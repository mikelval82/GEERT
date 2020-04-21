#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

from COM.trigger_client import trigger_client

tc = trigger_client('10.1.28.117',10000)
tc.create_socket()
tc.connect()
tc.send_msg(b'start')
tc.send_msg(b'1')
tc.send_msg(b'9')
tc.send_msg(b'5')
tc.send_msg(b'stop')