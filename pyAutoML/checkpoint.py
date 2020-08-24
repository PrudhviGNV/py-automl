import keras
from keras import model, layers


def save_model(model, verbose=True, config == True):
    """Saves the model in the disk. We can both save the architecture in JSON file and model in .h5 file

    Parameters:
    model(object): Giving the reference of the model
    verbose(bool): Tells and prints information of the activity
    config(bool): It tells function to save architecture of the model or not

    """
  if config == True:
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    # model.save_weights("model.h5")
    if verbose==True:
      print("Saved model architecture to disk")
  model.save("model.h5")

  
 
# later...
 
# load json and create model

def load_model(config=True):
    """
    Loads the model from memory to disk

    Parameters:
    config(bool): To load architecture of model or not

    Returns:
    reference of the loaded model
    """
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights("model.h5")
  return model
