import classifyImage as classify
import detectObjects as detect
import imageSegment as imSeg
import targetClassify


#%% Image Classification
    
# Print the prediction of the image classification to console (top5)
classify.classifyImage('ResNet50_v1d','mt_baker.jpg')

#%% Object Detection
    
# Detect objects in the image and plot with the boundaries of detected objects
detect.detectObjects('yolo3_darknet53_coco', 'dog.jpg')

#%% Image Segmentation
# Use same image as object detection
# Will output an image showing the different segments, colored
imSeg.imageSegment('fcn_resnet50_ade','dog.jpg')

#%% Specific object classification

# Output probability that a basketball appears in a selected image    
targetClassify.targetClassify('ResNet50_v1d','bball.jpg','basketball')
targetClassify.targetClassify('ResNet50_v1d','tennisball.jpg','basketball')
















        



    


