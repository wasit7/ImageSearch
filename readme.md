#Image classification project
Started from 24 th April 2014.

A python base image classification on ipython parallel.
Todo list
### ./App
1. make sure the app can be export to other machines <Flame>

### ./Randomforest/Client.py
1. Client need to access dataset by using function members <KK>
   getX() to random set of samples,  e.g samples=[1,2,4,5,6], use numpy array is possible (Off is doing on this)
   getSize() to get the size of  samples. this similar to len(samples)
   getL( x )
   getI( theta, x )
   getParam( x )
___

### class LibraryImageDataset
1. Implement getX() <Off>
   to random set of n coordinates form each image and store in this format,  samples = { [ r, c, img ] }
2. change the json write and load data structure <Flame>
   a data member in dataset class Limages (labeledImages) keeps all rectangles info,
   Limages[ image index ] [rect index]
3. getI( theta, x )
