**Emotion Recogintion with face detection** 😀😠☹️😩

As we all know nowadays computer vision is getting advanced.Facial Emotion Detection is a quite useful & emerging topic in computer vision .As its applications are quite useful for identification like driver’s drowsiness detection, students behavior detection, medical research in autism therapy and deepfake detection etc.
Such interesting applications have made facial expression recognition a hot research topic among deep learning engineers.

This project is further divided into the following steps :


-	Getting Data
-	Preparing data
-	Image Augmentation
-	Build model and train &	use the webcam for detection

Getting data : In this project dataset used is  fer-2013 which is publically available on Kaggle. it has 48*48 pixels gray-scale images of faces along with their emotion labels.This dataset contains 7 Emotions :- (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

Preparing Data : Here X_train, X_test contains pixels, and y_test, y_train contains emotions. At this stage X_train, X_test contains pixel’s number is in the form of a string, converting it into numbers is easy,just need to typecast.

Image Augmentation :Image data augmentation is used to improve the performance and ability of the model to generalize. It’s always a good practice to apply some
data augmentation before passing it to the model, which can be done using ImageDataGenetrator provided by Keras.

Building Facial Emotion Detection Model using CNN :Designing the CNN model for emotion detection using functional API. We are creating blocks using Conv2D layer, Batch-Normalization, Max-Pooling2D, Dropout, Flatten, and then stacking them together and at the end-use Dense Layer for output. Compiling & saving the model. using Haar-cascade for the detection position of faces and after getting position we will crop the faces & using openCv. 


