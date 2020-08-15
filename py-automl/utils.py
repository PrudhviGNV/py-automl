

def testImage(imagePath) :

    """

    Used to preprocess test image - resize, reshape, greyscale , normalise to feed as input to model for predictions

    Parameters:
    imagepath(str): passes the path of the image

    Returns:
    prediction
    """



  import cv2
  import numpy as np
  
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
  prediction = model.predict(cropped_img)

  maxindex = int(np.argmax(prediction))

  if img_show == True:
    cv2.imshow(image)
  return maxindex





def split(x ,y,verbose=True):

"""
Used to split dataset into training and validation dataset

Parameters:
x(list): the input dataset
y(list): output vector
verbose(bool): used to print information about dataset

Returns:
 tuple --> test and  validate dataset
 """
    import numpy
    import pandas

    num_samples, num_classes = y.shape

    num_samples = len(x)
    num_train_samples = int((1 - 0.2)*num_samples)

    # Traning data
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]

    # Validation data
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]

    train_data = (train_x, train_y)
    val_data = (val_x, val_y)

    if verbose==True:
        print('Training Pixels',train_x.shape)  # ==> 4 dims -  no of images , width , height , color
        print('Training labels',train_y.shape)

        print('Validation Pixels',val_x.shape)
        print('Validation labels',val_y.shape)

    return train_data,val_data