# On importe les librairies necéssaires 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])   #Définition du format des données 

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform) #Importation des échantillons d'entraînement
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)      #On définit la mise en forme des données 

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)  # Importation des échantillons de test 
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=2)      #On définit la mise en forme des données

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')        # On définit les classes possibles pour les entrées 

# fonction d'affichage des images


def imshow(img):
    img = img / 2 + 0.5    #On transforme l'image en image affichable par matplotlib
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# on prend des images aléatoirement
dataiter = iter(trainloader)
images, labels = dataiter.next()

# affichage des images
imshow(torchvision.utils.make_grid(images))	
# print leur classe 
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


### Création du réseau ### 
class Net(nn.Module):
    def __init__(self): # initialisation 
        super(Net, self).__init__() #initialisation de nn.Module  
        self.conv1 = nn.Conv2d(1, 10, 5) #Nuances de gris ==> 1 canal couleur, 10 canaux de sortie 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 14, 5)#Classes ==> 10, 14 car l'image a une résolution de 28*28
        self.fc1 = nn.Linear(14 * 4 * 4, 320)# définition du réseau 
        self.fc2 = nn.Linear(320, 50)
        self.fc3 = nn.Linear(50, 10) #Donnant 10 sorties possibles, tous les chiffres 0 --> 9

    def forward(self, x): # Fonction pour avancer dans le réseau 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 14*4*4) #On change le format de x 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net() # crée le réseau 


criterion = nn.CrossEntropyLoss() #On choisit une fonction de coût 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9) #descente de gradient


for epoch in range(2):  # On tourne deux fois sur le set de données 

    actual_loss = 0.0 
    for i, data in enumerate(trainloader, 0):
        # on récupère les inputs où data = [input, label]
        inputs, labels = data

        # réinitialiser les gradient 
        optimizer.zero_grad()
        #Parcours du réseau 
        outputs = net(inputs) # on passe les inputs dans le réseau
        loss = criterion(outputs, labels)#on calcule l'erreur 
        loss.backward()#backprop de l'erreur 
        optimizer.step()#actualisation des poids/biais 

        # affichage de l'époque, du nombre d'échantillons et de l'erreur 
        actual_loss += loss.item()
        if i % 2000 == 1999:    # tous les 2000 échantillons 
            print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, actual_loss / 2000))  
            actual_loss = 0.0

print('Finished Training')
PATH = './MNIST_net.pth'
torch.save(net.state_dict(), PATH) #sauvegarde du réseau 

#image de test aléatoires  
dataiter = iter(testloader)  
images, labels = dataiter.next() 

# affichage 
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


net = Net()
net.load_state_dict(torch.load(PATH)) # on charge le réseau 

outputs = net(images) # on passe les images de test dans le réseau 

_, predicted = torch.max(outputs, 1) #on prend les valeurs retournées 

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#Affichage de la précision globale 
correct = 0
total = 0
with torch.no_grad(): #en ne calculant pas les gradients 
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() 

print('Précision du réseau sur 10000 échantillons  : {} %'.format(
    100 * correct / total))

#Affichage de la précision pour chaque chiffre 
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():#en ne calculant pas les gradients 
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('précision {} : {} %'.format(
        classes[i], 100 * class_correct[i] / class_total[i]))