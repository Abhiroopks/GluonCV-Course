import gluoncv as gcv
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from gluoncv.utils.viz import get_color_pallete
import matplotlib.pyplot as plt

def imageSegment(model_name,input_pic):
    
    # Load Images
    img = mx.image.imread("images/" + input_pic)
    
    # Transform
    transform_fn = transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    img = transform_fn(img)
    
    # Add extra dimension to img for batch size
    img = img.expand_dims(0)
    
    # Assume pretrained
    net = gcv.model_zoo.get_model(model_name,pretrained=True)
    
    # Output has dimensions of: (# images, # classes, pixelY, pixelX) in logits
    out = net.predict(img)
    out = out[0]
    # pred will contain the class index with highest probability for each pixel
    pred = mx.nd.argmax(out,0).asnumpy()
    pred_img = get_color_pallete(pred,'ade20k')
    plt.figure()
    plt.imshow(pred_img)


