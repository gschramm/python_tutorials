import sys
import socket
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger, StoragePresentationContexts
import pydicom
from pydicom.dataset import FileDataset
from pathlib import Path
import shutil

from time import sleep
import threading

import numpy as np
from scipy.ndimage import binary_dilation

import pymirc.fileio as pf

from datetime import datetime

import logging
from logging.handlers import TimedRotatingFileHandler

#----------------------------------------------------------------------------------------------------------------
def setup_logger(log_path, level = logging.INFO, formatter = None, mode = 'a'):
  """ wrapper function to setup a file logger with some usefule properties (format, file replacement ...)
  """

  # create log file if it does not exist
  if not log_path.exists():
    log_path.touch() 

  if formatter is None:
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt = '%Y/%d/%m %I:%M:%S %p')

  logger = logging.getLogger()
  logger.setLevel(level)
  logger.handlers = []

  # log to file  
  handler = TimedRotatingFileHandler(filename = log_path, when = 'D', interval = 7, backupCount = 4, 
                                     encoding = 'utf-8', delay = False)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # log to stdout as well
  streamHandler = logging.StreamHandler(sys.stdout)
  streamHandler.setFormatter(formatter)
  logger.addHandler(streamHandler)

  return logger


#----------------------------------------------------------------------------------------------------------------
def send_dcm_files_to_server(dcm_file_list, dcm_server_ip, dcm_server_port, logger):
  # Initialise the Application Entity
  ae = AE()
  ae.requested_contexts = StoragePresentationContexts
  
  
  assoc = ae.associate(dcm_server_ip, dcm_server_port)
  if assoc.is_established:
    logger.info('Association established')
  
    # Use the C-STORE service to send the dataset
    # returns the response status as a pydicom Dataset

    for dcm_file in dcm_file_list:
      logger.info(f'sending {dcm_file}')
      dcm = pydicom.read_file(dcm_file)
      status = assoc.send_c_store(dcm, originator_aet = 'Fermi')
  
      # Check the status of the storage request
      if status:
        # If the storage request succeeded this will be 0x0000
        logger.info('C-STORE request status: 0x{0:04x}'.format(status.Status))
      else:
        logger.info('Connection timed out, was aborted or received invalid response')
  
    # Release the association
    assoc.release()
    logger.info('Association released')
  else:
    logger.info('Association rejected, aborted or never connected')


#----------------------------------------------------------------------------------------------------------------
def dummy_workflow_2(img_dcm_path, rtst_dcm_path, rtstruct_output_fname, sending_address, sending_port, logger):

  rtstruct_file = str(list(rtst_dcm_path.glob('*.dcm'))[0])
  
  # read the dicom volume
  dcm = pf.DicomVolume(str(Path(img_dcm_path / '*.dcm')))
  vol = dcm.get_data()
  
  # read the ROI contours (in world coordinates)
  contour_data = pf.read_rtstruct_contour_data(rtstruct_file)
  
  # convert contour data to index arrays (voxel space)
  roi_inds = pf.convert_contour_data_to_roi_indices(contour_data, dcm.affine, vol.shape)

  #---------------------------------------------------------------------------
  # create a label array
  roi_vol = np.zeros(vol.shape, dtype = np.uint16)
  
  for i in range(len(roi_inds)):
    roi_vol[roi_inds[i]] = int(contour_data[i]['ROINumber'])
  
  # create dilated ROI
  roi_vol2 = binary_dilation(roi_vol, iterations = 3)
  
  pf.labelvol_to_rtstruct(roi_vol2, dcm.affine, dcm.filelist[0], str(rtstruct_output_fname), 
                          tags_to_add = {'SpecificCharacterSet':'ISO_IR 192'})
  logger.info(f'wrote {rtstruct_output_fname}')

  # send new rtstruct back to server
  logger.info('sending RTstruct back')
  send_dcm_files_to_server([rtstruct_output_fname], sending_address, sending_port, logger)



#----------------------------------------------------------------------------------------------------------------
class DicomListener:
  def __init__(self, storage_dir = Path('.'), processing_dir = Path('.') / 'processing', 
                     sending_port = 104, cleanup_process_dir = True):
    self.storage_dir         = storage_dir.resolve()
    self.processing_dir      = processing_dir.resolve()
    self.sending_port        = sending_port 
    self.cleanup_process_dir = cleanup_process_dir

    self.logger = setup_logger(self.storage_dir / 'dicom_process.log')
    self.logger.info('intializing dicom processor')

    self.last_dcm_storage_dir = None
    self.last_dcm_fname       = None
    self.last_peer_address    = None
    self.last_peer_ae_tile    = None
    self.last_peer_port       = None
    self.last_ds              = None

  # Implement a handler for evt.EVT_C_STORE
  def handle_store(self,event):
    self.last_dcm_storage_dir = None
    self.last_dcm_fname       = None
    self.last_peer_address    = None
    self.last_peer_ae_tile    = None
    self.last_peer_port       = None
    self.last_ds              = None

    """Handle a C-STORE request event."""
  
    # get the IP of the sender
    assoc = threading.current_thread()
    self.last_peer_address = assoc.remote['address']
    self.last_peer_ae_tile = assoc.remote['ae_title']
    self.last_peer_port    = assoc.remote['port']
  
    # get string of series description and remove all non alpha-num characters
    sdesc = ''.join(filter(str.isalnum, event.dataset.SeriesDescription))
  
    self.last_dcm_storage_dir = self.storage_dir / f'{event.dataset.StudyDate}_{event.dataset.PatientID}_{event.dataset.StudyInstanceUID}' / f'{event.dataset.Modality}_{sdesc}_{event.dataset.SeriesInstanceUID}'
    self.last_dcm_storage_dir.mkdir(exist_ok = True, parents = True)
 
    self.last_dcm_fname = self.last_dcm_storage_dir / f'{event.dataset.SOPInstanceUID}.dcm'
 
    # Save the dataset using the SOP Instance UID as the filename
    self.last_ds = FileDataset(self.last_dcm_fname, event.dataset, file_meta = event.file_meta)

    self.last_ds.save_as(self.last_dcm_fname, write_like_original = False)
  
    # if Modality is RTstruct, save the referenced series UID as txt file
    if self.last_ds.Modality == 'RTSTRUCT':
      refSeriesUID = self.last_ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
      (self.last_dcm_storage_dir / f'{refSeriesUID}.rtxt').touch()

    # Return a 'Success' status
    return 0x0000
  
  def handle_accepted(self,event):
    self.logger.info('accepted')
  
  def handle_released(self,event):
    self.logger.info('released')
    # start processing here
    if self.last_dcm_storage_dir is not None:
      self.logger.info('')
      self.logger.info(f'series desc  ..: {self.last_ds.SeriesDescription}')
      self.logger.info(f'series UID   ..: {self.last_ds.SeriesInstanceUID}')
      self.logger.info(f'modality     ..: {self.last_ds.Modality}')
      self.logger.info(f'storage dir  ..: {self.last_dcm_storage_dir}')
      self.logger.info(f'peer address ..: {self.last_peer_address}')
      self.logger.info(f'peer AE      ..: {self.last_peer_ae_tile}')
      self.logger.info(f'peer port    ..: {self.last_peer_port}')    
      self.logger.info('')

      # if the incoming dicom data is CT or MR, we check wether an RTstruct defined
      # on that series exist 

      if self.last_ds.Modality == 'CT' or self.last_ds.Modality == 'MR':
        rtxt_files = list(self.last_dcm_storage_dir.parent.rglob(f'{self.last_ds.SeriesInstanceUID}.rtxt'))

        if len(rtxt_files) == 0:
          # no corresponding RTstruct file exists, run workflow 1
          self.logger.info('running workflow 1')
          self.logger.info(f'image input {self.last_dcm_storage_dir}')
        else:
          # corresponding RTstruct file exists, run workflow 2
          self.logger.info('running workflow 2')
          try:
            self.logger.info(f'image    input {self.last_dcm_storage_dir}')
            self.logger.info(f'RTstruct input {rtxt_files[0].parent}')

            # move input dicom series into process dir
            process_dir = self.processing_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self.last_dcm_storage_dir.parent.name}'
            process_dir.mkdir(exist_ok = True, parents = True)
         
            shutil.move(self.last_dcm_storage_dir, process_dir / 'image') 
            self.logger.info(f'moving {self.last_dcm_storage_dir} to {process_dir / "image"}')
            shutil.move(rtxt_files[0].parent, process_dir / 'rtstruct') 
            self.logger.info(f'moving {rtxt_files[0].parent} to {process_dir / "rtstruct"}') 

            # check if study dir is empty after moving series, and delete if it is
            if not any(Path(self.last_dcm_storage_dir.parent).iterdir()):
              shutil.rmtree(self.last_dcm_storage_dir.parent)
              self.logger.info(f'removed empty dir {self.last_dcm_storage_dir.parent}')

            # run dummy workflow to create new RTstruct
            output_rstruct_fname = process_dir / 'dummy_rtstruct_2.dcm'
            dummy_workflow_2(process_dir / 'image', process_dir / 'rtstruct', output_rstruct_fname,
                             self.last_peer_address, self.sending_port, self.logger)
            
            if self.cleanup_process_dir:
              shutil.rmtree(process_dir)
              self.logger.info(f'removed {process_dir}')
          except:
            self.logger.error('workflow 2 failed')  

      #if self.last_ds.Modality == 'DOC':
      #  self.logger.info(self.last_ds.EncapsulatedDocument.decode("utf-8"))
  
  def handle_echo(self,event):
    self.logger.info('echo')

#------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  dcm_listener = DicomListener(Path.home() / 'tmp' / 'dcm', processing_dir = Path.home() / 'tmp' / 'process')
  
  handlers = [(evt.EVT_C_STORE,  dcm_listener.handle_store), 
              (evt.EVT_RELEASED, dcm_listener.handle_released),
              (evt.EVT_ACCEPTED, dcm_listener.handle_accepted),
              (evt.EVT_C_ECHO,   dcm_listener.handle_echo)]
  
  # Initialise the Application Entity
  ae = AE('FERMI-TEST')
  
  # Support presentation contexts for all storage SOP Classes
  ae.supported_contexts = AllStoragePresentationContexts
  
  # Start listening for incoming association requests
  ae.start_server((socket.gethostbyname(socket.gethostname()), 11112), evt_handlers = handlers, block = True)
