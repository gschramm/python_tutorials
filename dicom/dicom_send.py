import pydicom
from pynetdicom import AE, debug_logger, StoragePresentationContexts

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

#-------------------------------------------------------------------------

if __name__ == '__main__':
  #debug_logger()
  
  # read IP adress and port of dicom server where we send data to
  with open('.config', 'r') as f:
    cfg = f.read().splitlines()
  
  # read test data set from pydicom
  #ds = pydicom.read_file(pydicom.data.get_testdata_files('MR2_UNCR.dcm')[0])
  dcm_files = pydicom.data.get_testdata_files('MR2_UNC*.dcm')
  
  send_dcm_files_to_server(dcm_files, str(cfg[0]), int(cfg[1]))
