from mxnet import image

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data.transforms.presets.yolo import transform_test
from gluoncv.model_zoo import get_model

def detectObjects(model_name,input_pic):
  
    # Load Model, assume pretrained
    
    net = get_model(model_name, pretrained=True)
    
    # Load Images
    img = image.imread("images/" + input_pic)
    
    # Transform
    img, chw_im = transform_test(img)
    pred = net(img)
    
    # Assume only one image in batch, use first array only
    pred = [array[0] for array in pred]
    
    # Unpack tuple into each array
    class_ind, prob, bounds = pred
    
    gcv.utils.viz.plot_bbox(chw_im,bounds,prob,class_ind,class_names=net.classes)
    
    
