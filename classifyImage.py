from mxnet import nd, image

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model

def classifyImage(model_name,input_pic):
  
    # Load Model
    # Assume pretrained
    net = get_model(model_name, pretrained=True)
    
    classes = net.classes
    
    # Load Images
    img = image.imread("images/" + input_pic)
    
    # Transform
    img = transform_eval(img)
    pred = net(img)
    
    # Display the top 5 classes from prediction
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input picture is classified to be')
    for i in range(topK):
        print('\t[%s], with probability %.3f.'%
              (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
