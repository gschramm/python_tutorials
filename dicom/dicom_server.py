import socket
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger
from pydicom.dataset import FileDataset
from pathlib import Path

from time import sleep

import threading

#debug_logger()

class DicomListener:
  def __init__(self, storage_dir = Path('.')):
    self.last_dcm_storage_dir = None
    self.last_dcm_fname       = None
    self.last_peer_address    = None
    self.last_peer_ae_tile    = None
    self.last_peer_port       = None
    self.last_ds              = None
    self.storage_dir          = storage_dir.resolve()


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
 
      #if self.last_ds.Modality == 'DOC':
      #  print(self.last_ds.EncapsulatedDocument.decode("utf-8"))
  
  def handle_echo(self,event):
    print('XXXXXX echo')

#------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  dcm_listener = DicomListener(Path.home() / 'tmp' / 'dcm')
  
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
