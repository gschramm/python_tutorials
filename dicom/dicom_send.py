import pydicom
from pynetdicom import AE, debug_logger, StoragePresentationContexts

#debug_logger()

# read IP adress and port of dicom server where we send data to
with open('.config', 'r') as f:
  cfg = f.read().splitlines()

# Initialise the Application Entity
ae = AE()
ae.requested_contexts = StoragePresentationContexts

# read test data set from pydicom
#ds = pydicom.read_file(pydicom.data.get_testdata_files('MR2_UNCR.dcm')[0])
dcm_file = pydicom.data.get_testdata_files('MR2_UNCR.dcm')[0]

assoc = ae.associate(str(cfg[0]), int(cfg[1]))
if assoc.is_established:
  print('Association established')

  # Use the C-STORE service to send the dataset
  # returns the response status as a pydicom Dataset
  status = assoc.send_c_store(dcm_file, originator_aet = 'Fermi')

  # Check the status of the storage request
  if status:
    # If the storage request succeeded this will be 0x0000
    print('C-STORE request status: 0x{0:04x}'.format(status.Status))
  else:
    print('Connection timed out, was aborted or received invalid response')

  # Release the association
  assoc.release()
  print('Association released')
else:
  print('Association rejected, aborted or never connected')
