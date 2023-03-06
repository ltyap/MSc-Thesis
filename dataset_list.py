from dataset_specifications.aero import AeroSet
from dataset_specifications.aero_MDN import AeroMDNSet
from dataset_specifications.aerohydro import AeroHydroSet


set = {
    'aero': AeroSet,
    'aero_MDN': AeroMDNSet,
    'aerohydro': AeroHydroSet
}

def get_dataset_spec(name):
    return set[name]