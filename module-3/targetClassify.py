from mxnet import nd, image

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model



def targetClassify(model_name,input_pic,target_class):
    # The purpose of this is to simply output the percent probability that
    # the image is specified target_class
    # Load specified Model
    # Assume pretrained
    net = get_model(model_name, pretrained=True)
    
    classes = net.classes
    
    classInd = -1;
    # Find index of target class
    for i,j in enumerate(classes):
        if target_class == j.lower():
            classInd = i
            break
        
    # Exit if target class not found
    if classInd == -1:
        print("ERROR: Target class not found in this model : %s" % target_class)
        return            
    
    # Load Images, assume all data is in "images/" directory
    img = image.imread("images/" + input_pic)
    
    # Transform and predict
    img = transform_eval(img)
    pred = net(img)
    # use softmax and print probability
    #prob = nd.softmax(pred)
    print("Probability of class [%s] for [%s]: %.3f" % (classes[classInd],input_pic,nd.softmax(pred)[0][classInd].asscalar()))   
    