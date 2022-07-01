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

#----------------------------------------------------------------------------------------------------------------
def send_dcm_files_to_server(dcm_file_list, dcm_server_ip, dcm_server_port, verbose = True):
  # Initialise the Application Entity
  ae = AE()
  ae.requested_contexts = StoragePresentationContexts
  
  
  assoc = ae.associate(dcm_server_ip, dcm_server_port)
  if assoc.is_established:
    if verbose: print('Association established')
  
    # Use the C-STORE service to send the dataset
    # returns the response status as a pydicom Dataset

    for dcm_file in dcm_file_list:
      if verbose: print(dcm_file)
      dcm = pydicom.read_file(dcm_file)
      status = assoc.send_c_store(dcm, originator_aet = 'Fermi')
  
      # Check the status of the storage request
      if status:
        # If the storage request succeeded this will be 0x0000
        if verbose: print('C-STORE request status: 0x{0:04x}'.format(status.Status))
      else:
        if verbose: print('Connection timed out, was aborted or received invalid response')
  
    # Release the association
    assoc.release()
    if verbose: print('Association released')
  else:
    if verbose: print('Association rejected, aborted or never connected')


#----------------------------------------------------------------------------------------------------------------
def dummy_workflow_2(img_dcm_path, rtst_dcm_path, rtstruct_output_fname, sending_address, sending_port):
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
  print(f'wrote {rtstruct_output_fname}')

  # send new rtstruct back to server
  print('sending RTstruct back')
  send_dcm_files_to_server([rtstruct_output_fname], sending_address, sending_port, verbose = True)



#----------------------------------------------------------------------------------------------------------------
class DicomListener:
  def __init__(self, storage_dir = Path('.'), processing_dir = Path('.') / 'processing', 
                     sending_port = 104, cleanup_process_dir = True):
    self.storage_dir         = storage_dir.resolve()
    self.processing_dir      = processing_dir.resolve()
    self.sending_port        = sending_port 
    self.cleanup_process_dir = cleanup_process_dir


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
    print('XXXXXX accepted')
  
  def handle_released(self,event):
    print('XXXXXX released')
    # start processing here
    if self.last_dcm_storage_dir is not None:
      print('')
      print('series desc  ..:', self.last_ds.SeriesDescription)
      print('series UID   ..:', self.last_ds.SeriesInstanceUID)
      print('modality     ..:', self.last_ds.Modality)
      print('storage dir  ..:', self.last_dcm_storage_dir)
      print('peer address ..:', self.last_peer_address)
      print('peer AE      ..:', self.last_peer_ae_tile)
      print('peer port    ..:', self.last_peer_port)    
      print('')

      # if the incoming dicom data is CT or MR, we check wether an RTstruct defined
      # on that series exist 

      if self.last_ds.Modality == 'CT' or self.last_ds.Modality == 'MR':
        rtxt_files = list(self.last_dcm_storage_dir.parent.rglob(f'{self.last_ds.SeriesInstanceUID}.rtxt'))

        if len(rtxt_files) == 0:
          # no corresponding RTstruct file exists, run workflow 1
          print('running workflow 1')
          print('image    input', self.last_dcm_storage_dir)
        else:
          # corresponding RTstruct file exists, run workflow 2
          print('running workflow 2')
          print('image    input', self.last_dcm_storage_dir)
          print('RTstruct input', rtxt_files[0].parent)

          # move input dicom series into process dir
          process_dir = self.processing_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self.last_dcm_storage_dir.parent.name}'
          process_dir.mkdir(exist_ok = True, parents = True)
         
          shutil.move(self.last_dcm_storage_dir, process_dir / 'image') 
          shutil.move(rtxt_files[0].parent, process_dir / 'rtstruct') 

          # check if study dir is empty after moving series, and delete if it is
          if not any(Path(self.last_dcm_storage_dir.parent).iterdir()):
            shutil.rmtree(self.last_dcm_storage_dir.parent)

          # run dummy workflow to create new RTstruct
          output_rstruct_fname = process_dir / 'dummy_rtstruct_2.dcm'
          dummy_workflow_2(process_dir / 'image', process_dir / 'rtstruct', output_rstruct_fname,
                           self.last_peer_address, self.sending_port)
          
          if self.cleanup_process_dir:
            shutil.rmtree(process_dir)

      #if self.last_ds.Modality == 'DOC':
      #  print(self.last_ds.EncapsulatedDocument.decode("utf-8"))
  
  def handle_echo(self,event):
    print('XXXXXX echo')

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
