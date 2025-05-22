from utils.utils_model import *
from utils.utils_params import *

DATA_PATH = "data"
MODEL_PATH = "models"

search_space_map = {
    "LR": get_LR_search_space,
    "RF": get_RF_search_space,
    "XGB": get_XGB_search_space
}

const_param_map = {
    "LR": get_LR_const_params,
    "RF": get_RF_const_params,
    "XGB": get_XGB_const_params
}

trainer_map = {
    'LR': train_LR_model,
    'RF': train_RF_model,
    'XGB': train_XGB_model
}
