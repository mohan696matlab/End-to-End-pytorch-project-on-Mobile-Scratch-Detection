import torch, torchvision
from torchvision import transforms

from PIL import Image

# Makes the code device agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classes = ['oil', 'scratch', 'stain']

weights = torchvision.models.MobileNet_V2_Weights
cnn_model = torchvision.models.mobilenet_v2(weights = weights.DEFAULT) 
for param in cnn_model.parameters():
    param.requires_grad = False
cnn_model.classifier[1] = torch.nn.Linear(in_features= cnn_model.classifier[1].in_features,
                                          out_features=len(classes))
ww = torch.load('cnn_model_1.pth')
cnn_model.load_state_dict(ww)
cnn_model.to(device)
cnn_model.eval()


transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

def predict_screen_defect(image):
    image = transform(Image.open(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(image)

    lable_pred = classes[outputs.argmax().item()]
    pred_confidence = outputs.softmax(dim=1).max().item()
    
    return lable_pred, pred_confidence
