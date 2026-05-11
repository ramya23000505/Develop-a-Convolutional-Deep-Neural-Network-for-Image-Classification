# Develop a Convolutional Deep Neural Network for Image Classification
### DATE : 11-05-2026
## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.

1.Training data: 60,000 images

2.Test data: 10,000 images

3.Classes: 10 fashion categories

The CNN consists of multiple convolutional layers with activation functions, followed by pooling layers, and ends with fully connected layers to output predictions for all 10 categories.


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1: 

Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 

Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 

Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 

Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 

Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 

Evaluate the trained model on test images and verify the classification accuracy for new unseen images.


## PROGRAM

### Name: RAMYA R

### Register Number: 212223230169

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # write your code here
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
model =
criterion =
optimizer =

# Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here

        
        
        
        print('Name:        ')
        print('Register Number:       ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

Include the Training Loss per epoch

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
