import torch.optim as optim
import torchvision.models as models

from modulesGramMatrix import *

class StyleCNN(object):
    def __init__(self, style, content, pastiche):
        super(StyleCNN, self).__init__()
        
        self.style = style
        self.content = content
        self.pastiche = nn.Parameter(pastiche.data)
        
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000
        
        self.loss_network = models.vgg19(pretrained=True)
        
        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.LBFGS([self.pastiche])
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()
        self.transform_network = nn.Sequential(nn.ReflectionPad2d(40),
                                           nn.Conv2d(3, 32, 9, stride=1, padding=4),
                                           nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                           nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                           nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                           nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                           nn.Conv2d(32, 3, 9, stride=1, padding=4),
                                           )
        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.transform_network.parameters(), lr=1e-3)
    
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()

    def train(self):
        def closure():
            self.optimizer.zero_grad()
          
            pastiche = self.pastiche.clone()
            pastiche.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()
            
            content_loss = 0
            style_loss = 0
            
            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()
                    
                pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)
                
                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i)
                    
                    if name in self.content_layers:
                        content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)
                    
                    if name in self.style_layers:
                        pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                        style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)
                
                if isinstance(layer, nn.ReLU):
                    i += 1
            
            total_loss = content_loss + style_loss
            total_loss.backward()
            
            return total_loss
    
        self.optimizer.step(closure)
        return self.pastiche


import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import scipy.misc

imsize = 256

loader = transforms.Compose([
             transforms.Scale(imsize),
             transforms.ToTensor()
         ])

unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image
  
def save_image(input, path):
    image = input.data.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    scipy.misc.imsave(path, image)

import torch.utils.data
import torchvision.datasets as datasets


from utils import *

# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Content and style
style = image_loader("styles/starry_night.jpg").type(dtype)
content = image_loader("contents/1212.jpeg").type(dtype)

pastiche = image_loader("contents/1212.jpeg").type(dtype)
pastiche.data = torch.randn(input.data.size()).type(dtype)

num_epochs = 31

def main():
    style_cnn = StyleCNN(style, content, pastiche)
    
    for i in range(num_epochs):
        pastiche = style_cnn.train()
    
        if i % 10 == 0:
            print("Iteration: %d" % (i))
            
            path = "outputs/%d.png" % (i)
            pastiche.data.clamp_(0, 1)
            save_image(pastiche, path)

main()

