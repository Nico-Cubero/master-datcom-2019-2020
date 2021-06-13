# Inicializar funciones y clases
from eci.utils import (load_image_class_dataset, compute_HOG, compute_basic_LBP,
					  compute_uniform_LBP, compute_HOG_LBP,
					  check_experiment_document, extract_experiments_parameters)
from eci.svm_functions import train_svm, test_svm
from eci.lbp import LBPDescriptor, UniformLBPDescriptor