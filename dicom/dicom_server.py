import socket
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger
from pathlib import Path

from time import sleep

debug_logger()

# Implement a handler for evt.EVT_C_STORE
def handle_store(event):
    """Handle a C-STORE request event."""

    # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
    ds = event.dataset

    # Add the File Meta Information
    ds.file_meta = event.file_meta

    odir = Path(ds.SeriesInstanceUID)
    odir.mkdir(exist_ok = True, parents = True)

    # Save the dataset using the SOP Instance UID as the filename
    ds.save_as(odir / f'{ds.SOPInstanceUID}.dcm', write_like_original = True)

    # Return a 'Success' status
    return 0x0000

def handle_released(event):
  print('XXXXXX released')

handlers = [(evt.EVT_C_STORE, handle_store), (evt.EVT_RELEASED, handle_released)]

# Initialise the Application Entity
ae = AE('FERMI-TEST')

# Support presentation contexts for all storage SOP Classes
ae.supported_contexts = AllStoragePresentationContexts

# Start listening for incoming association requests
ae.start_server((socket.gethostbyname(socket.gethostname()), 11112), evt_handlers = handlers)
