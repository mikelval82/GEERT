# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""

from COM.OpenBCISample import OpenBCISample

from multiprocessing import Process
import serial 
import struct
import time
import atexit
import logging
import threading
import sys
import glob

#%% CONSTANTS
SAMPLE_RATE = 250.0  # Hz Frecuencia de muestreo.
START_BYTE = 0xA0  # Byte que indica el inicio de un paquete de datos
END_BYTE = 0xC0  # Byte que indica el fianl de un paquete de datos


class OpenBCIBoard(Process):
    def __init__(self, queue, streaming, isconnected, log, port=None, baud=115200, filter_data=True, timeout=1):
    
        """
        Se encarga de realizar la conexión con la tarjeta OpenBci.
        Define las funciones para desempaquetar los datos, comunicarse
        con la tarjeta, iniciar y detener el envío de datos.
        
        """
        Process.__init__(self)  
        #initializations
        self.baudrate = baud
        self.timeout = timeout
        self.port = port     
        self.filtering_data = filter_data    
        self.eeg_channels_per_sample = 8 # de la tarjeta      
        self.aux_channels_per_sample = 3 # de la tarjeta
        self.read_state = 0
        self.attempt_reconnect = False
        self.last_reconnect = 0
        self.reconnect_freq = 5
        self.packets_dropped = 0
        self.scaling_output = True
        self.log = True
        self.log_packet_count = 0
        self.scale_fac_uVolts_per_count = 2.23517444553071e-08
        self.scale_fac_accel_G_per_count = 0.002
        self.queue = queue
        self.streaming = streaming
        self.isconnected = isconnected
        self.log = log

        # Disconnects from board when terminated
        atexit.register(self.disconnect)
        
        try:
            if not self.port:
                self.port = self.find_port()
            print("Connecting to V3 at port %s" %(self.port))
        except:
            if not self.port:
                self.port = self.find_port()
            print("Connecting to V3 at port %s" %(self.port))
            
        self.ser = serial.Serial(port= self.port, baudrate = self.baudrate, timeout=self.timeout)
        print("Serial established...")  
       
    def run(self):       
        while True:  
            if self.streaming.value:
                try:
                    sample = self._read_serial_binary()
                    self.queue.put(sample.channel_data)
                except:
                    print('Driver Error while reading serial binaries')
            else:
                while not self.queue.empty():  
                    self.queue.get()

    def send_stop(self):
        self.streaming.value = False
        self.ser.write(b's')      
        self.log.update_text('Streaming: ' + str(self.streaming.value))
        
    def send_start(self):    
        self.streaming.value = True
        self.ser.write(b'b')
        self.check_connection()
        self.log.update_text('Streaming: ' + str(self.streaming.value))
             
    def connect(self):
        self.log.update_text("Connecting to V3 at port %s" %(self.port))
        self.ser = serial.Serial(port= self.port, baudrate = self.baudrate, timeout=self.timeout)
        self.log.update_text("Serial established...")  
        time.sleep(2)      
        self.log.update_text(str(self.ser.write(b'v')))
        time.sleep(1)
        self.print_incoming_text()
        self.isconnected.value = True
        
    def disconnect(self):
        #self.send_stop()
        
        if self.ser.isOpen():
          self.ser.close()
          self.log.update_text('Serial closed')
          self.isconnected.value = False          
          
    def enable_filters(self):
        '''
        ##  Activa los filtros internos de la tarjeta
        #   le dice a la tarjeta que active los filtros internos que tiene.
        '''
        self.ser.write(b'f')
        self.filtering_data = True;

    def disable_filters(self):
        '''
        Desctiva los filtros internos de la tarjeta
        '''
        self.ser.write(b'g')
        self.filtering_data = False;
#%%        
    def read(self, n):
            bb = self.ser.read(n)
            if not bb:
                print('Device appears to be stalled. Quitting...')
                sys.exit()
                raise Exception('Device Stalled')
                sys.exit()
                return '\xFF'
            else:
                return bb
            
    def _read_serial_binary(self, max_bytes_to_skip=3000):
        for rep in range(max_bytes_to_skip):     
            # ---------Start Byte & ID---------
            if self.read_state == 0: 
                b = self.read(1) 
   
                if struct.unpack('B', b)[0] == START_BYTE:
                    if (rep != 0):
                        print('Skipped %d bytes before start found' % (rep))
                        rep = 0
                    # packet id goes from 0-255
                    packet_id = struct.unpack('B', self.read(1))[0]
                    log_bytes_in = str(packet_id)
        
                    self.read_state = 1 
                
            # ---------Channel Data---------
            elif self.read_state == 1:
                channel_data = []
                for c in range(self.eeg_channels_per_sample):
                    # 3 byte ints
                    literal_read = self.read(3)
                    unpacked = struct.unpack('3B', literal_read)
                    log_bytes_in = log_bytes_in + '|' + str(literal_read)
                    # 3byte int in 2s compliment
                    if (unpacked[0] > 127):
                        pre_fix = bytes(bytearray.fromhex('FF'))
                    else:
                        pre_fix = bytes(bytearray.fromhex('00'))
                    literal_read = pre_fix + literal_read
                    myInt = struct.unpack('>i', literal_read)[0]     
                    if self.scaling_output:
                        channel_data.append(myInt * self.scale_fac_uVolts_per_count) # 
                    else:
                        channel_data.append(myInt)  
                    
                self.read_state = 2   
                
            # ---------Accelerometer Data---------
            elif self.read_state == 2:
                aux_data = []
                for a in range(self.aux_channels_per_sample):
        
                    # short = h
                    acc = struct.unpack('>h', self.read(2))[0]
                    log_bytes_in = log_bytes_in + '|' + str(acc)
        
                    if self.scaling_output:
                        aux_data.append(acc * self.scale_fac_accel_G_per_count)
                    else:
                        aux_data.append(acc)        
                self.read_state = 3
            # ---------End Byte---------
            elif self.read_state == 3:
                val = struct.unpack('B', self.read(1))[0]
                log_bytes_in = log_bytes_in + '|' + str(val)
                self.read_state = 0  # read next packet
                if (val == END_BYTE):
                    sample = OpenBCISample(packet_id, channel_data, aux_data)
                    self.packets_dropped = 0
                    return sample
                else:
#                    self.warn("ID:<%d> <Unexpected END_BYTE found <%s> instead of <%s>"
#                              % (packet_id, val, END_BYTE))
                    logging.debug(log_bytes_in)
                    self.packets_dropped = self.packets_dropped + 1
   
    def warn(self, text):
        print("Warning: %s" % text)

    def print_incoming_text(self):
        line = ''
        # Espera a que el dispositivo envíe datos
        time.sleep(1)
        
        if self.ser.inWaiting():
          line = ''
          c = ''
         # Busca la secuencia de fin '$$$'.
          while '$$$' not in line:
            c = self.ser.read().decode('utf-8')
            line += c
          print(line);
        else:
          self.warn("No Message")

    def print_register_settings(self):
        '''
        ##  Muestra las configuraciones registradas por la tarjeta
        #   Le dice a la tarjeta que envíe la información sobre las configuraciones registradas y la imprime con "print_incoming_text".
        #   @param  . No requiere parámetros.
        ##  @retval . No devuelve ningún valor.
        '''
        self.ser.write(b'?')
        time.sleep(0.5)
        self.print_incoming_text()

    def check_connection(self, interval = 2, max_packets_to_skip=10):
        if self.packets_dropped > max_packets_to_skip:
          self.reconnect()
        threading.Timer(interval, self.check_connection).start()

    def reconnect(self):
        self.packets_dropped = 0
        print('Reconnecting')
        self.stop()
        time.sleep(0.5)
        self.ser.write(b'v')
        time.sleep(0.5)
        self.ser.write(b'b')
        time.sleep(0.5)

    def find_port(self):
        print('Searching Board...')
        # nombre de los puertos seriales
        if sys.platform.startswith('win'):
          ports = ['COM%s' % (i+1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
          ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
          ports = glob.glob('/dev/tty.usbserial*')
        else:
          print('Error finding ports on your operating system')
        openbci_port = ''
        for port in ports:
            try:
                s = serial.Serial(port= port, baudrate = self.baudrate, timeout=self.timeout)

                if s.write(b'v')==1:
                  openbci_port = port;
                s.close()
            except:
                pass
        if openbci_port == '':      
            print('Cannot find OpenBCI port. Try again!!')
            return False
        else:
          return openbci_port

    def openbci_id(self, serial):
      	line = ''
      	# Espera a que el dispositivo envíe datos
      	time.sleep(2)
    
      	if serial.inWaiting():
      	  line = ''
      	  c = ''
      	 # Busca la secuencia de fin '$$$'
      	  while '$$$' not in line:
      	    c = serial.read().decode('utf-8')
      	    line += c
      	  if "OpenBCI" in line:
      	    return True
      	return False

