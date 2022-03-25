import socket
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger
from pathlib import Path

from time import sleep

#debug_logger()

# Implement a handler for evt.EVT_C_STORE
def handle_store(event):
    """Handle a C-STORE request event."""

    # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
    ds = event.dataset

    # Add the File Meta Information
    ds.file_meta = event.file_meta

    # get string of series description and remove all non alpha-num characters
    sdesc = ''.join(filter(str.isalnum, ds.SeriesDescription))

    odir = Path(f'{ds.StudyDate}_{ds.PatientID}_{ds.StudyInstanceUID}') / f'{ds.Modality}_{sdesc}_{ds.SeriesInstanceUID}'
    odir.mkdir(exist_ok = True, parents = True)

    # Save the dataset using the SOP Instance UID as the filename
    ds.save_as(odir / f'{ds.SOPInstanceUID}.dcm', write_like_original = True)

    # Return a 'Success' status
    return 0x0000

def handle_accepted(event):
  print('XXXXXX accepted')

def handle_released(event):
  print('XXXXXX released')

def handle_echo(event):
  print('XXXXXX echo')


#------------------------------------------------------------------------------------------------

handlers = [(evt.EVT_C_STORE, handle_store), 
            (evt.EVT_RELEASED, handle_released),
            (evt.EVT_ACCEPTED, handle_accepted),
            (evt.EVT_C_ECHO, handle_echo)]

# Initialise the Application Entity
ae = AE('FERMI-TEST')

# Support presentation contexts for all storage SOP Classes
ae.supported_contexts = AllStoragePresentationContexts

# Start listening for incoming association requests
ae.start_server((socket.gethostbyname(socket.gethostname()), 11112), evt_handlers = handlers, block = True)
