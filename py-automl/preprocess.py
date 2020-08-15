def image_preprocess(dataframe):
    """
    used to preprocess image dataset to resize , rescale , convert to greyscale, normalisation to make it suitable for training

    Parameters:
    dataframe: image in dataframe

    Returns :
    preprocessed image dataset

    """

  pixels = df['pixels'].tolist() # Converting the relevant column element into a list for each row
  width, height = 48, 48
  images = []

  for pixel_sequence in pixels:
    img = [int(pixel) for pixel in pixel_sequence.split(' ')] # Splitting the string by space character as a list
    img = np.asarray(face).reshape(width, height) #converting the list to numpy array in size of 48*48
    img = cv2.resize(face.astype('uint8'),image_size) #resize the image to have 48 cols (width) and 48 rows (height)
    images.append(face.astype('float32')) #makes the list of each images of 48*48 and their pixels in numpyarray form
    
  images = np.asarray(images) #converting the list into numpy array
  images = np.expand_dims(images, -1) #Expand the shape of an array -1=last dimension => means color space


  x = images.astype('float32')
  x = x / 255.0 #Dividing the pixels by 255 for normalization  => range(0,1)

  # Scaling the pixels value in range(-1,1)
  x = x - 0.5
  x = x * 2.0

  return x
