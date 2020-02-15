# Essentials
import numpy as np

# Model
import xgboost as xgb

# Misc
import argparse
import json

# Defining the parser
parser = argparse.ArgumentParser(description='Forecast rent price.')

parser.add_argument("-n", "--neighborhood", type=str, nargs=1, 
                    metavar = "neighborhood", default = None, 
                    help = "The neighborhood to check for apartements.") 
      
parser.add_argument("-b", "--beds", type=int, nargs=1, 
                    metavar = "beds", default = None, 
                    help = "The number of beds.") 
      
parser.add_argument("-B", "--baths", type = float, nargs = 1, 
                    metavar = "baths", default = None, 
                    help = "The number of baths.")

args = parser.parse_args()

# Loading the encoding dictionary
with open('dict.json') as json_file:
    neighbor = json.load(json_file)

# Load model
bst = xgb.Booster({'nthread': -1})
bst.load_model('model.bin')

# Print prediction
data = np.array([[neighbor['neighborhood'][args.neighborhood[0]], args.beds[0], args.baths[0]]])
print('An appartement with these caracteristics would cost {0:.2f}$.'.format(int(np.expm1(bst.predict(xgb.DMatrix(data))))))