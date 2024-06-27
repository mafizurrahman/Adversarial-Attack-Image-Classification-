import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
SEED = 1
N_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_CLASSES = 10  # CIFAR-10 has 10 classes

# Set the seed for reproducibility
torch.manual_seed(SEED)

# Load and transform the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit ResNet-50 input dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust if needed
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Define the ResNet-50 model
model = models.resnet50(pretrained=True)
# Modify the final layer to output the desired number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, N_CLASSES)

# Move the model to the appropriate device
model = model.to(DEVICE)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training function
def train(model, train_loader, criterion, optimizer, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
    print('Finished Training')

# Testing function
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the test images: {accuracy:.2f} %')
    return accuracy

# Train the model
train(model, train_loader, criterion, optimizer, N_EPOCHS, DEVICE)

# Test the model
test_accuracy = test(model, test_loader, DEVICE)

# Save the model
torch.save(model.state_dict(), 'resnet50_cifar10.pth')

# FGSM attack function
def fgsm_attack(model, loss, images, labels, epsilon, device):
    images = images.to(device)
    labels = labels.to(device)

    # Set requires_grad attribute of tensor to track the gradients
    images.requires_grad = True

    # Forward pass the data through the model
    outputs = model(images)
    model.zero_grad()
    cost = loss(outputs, labels).to(device)

    # Backward pass to get the gradients
    cost.backward()

    # Collect the element-wise sign of the data gradient
    attack_images = images + epsilon * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images

# Parameters for FGSM
EPSILON = 0.1

# Evaluate model accuracy on adversarial examples
def test_fgsm(model, test_loader, epsilon, criterion, device):
    correct = 0
    total = 0
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # Generate adversarial examples
        adv_images = fgsm_attack(model, criterion, images, labels, epsilon, device)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on FGSM adversarial test images (epsilon = {epsilon}): {accuracy:.2f} %')
    return accuracy

# Test the model against FGSM attack
test_fgsm_accuracy = test_fgsm(model, test_loader, EPSILON, criterion, DEVICE)


# Define the distillation model
class DistillModel(nn.Module):
    def __init__(self, teacher_model, num_classes, temperature):
        super(DistillModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = models.resnet50(pretrained=True)
        num_ftrs = self.student_model.fc.in_features
        self.student_model.fc = nn.Linear(num_ftrs, num_classes)
        self.temperature = temperature

    def forward(self, x):
        teacher_logits = self.teacher_model(x)
        student_logits = self.student_model(x)
        return student_logits, teacher_logits

# Distillation loss function
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    student_loss = F.cross_entropy(student_logits, labels)
    distill_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                            F.softmax(teacher_logits / temperature, dim=1),
                            reduction='batchmean') * (temperature ** 2)
    return alpha * student_loss + (1 - alpha) * distill_loss

# Parameters for distillation
TEMPERATURE = 3.0
ALPHA = 0.7

# Load the pretrained teacher model
teacher_model = models.resnet50(pretrained=True)
num_ftrs = teacher_model.fc.in_features
teacher_model.fc = nn.Linear(num_ftrs, N_CLASSES)
teacher_model.load_state_dict(torch.load('resnet50_cifar10.pth'))
teacher_model = teacher_model.to(DEVICE)
teacher_model.eval()

# Initialize the distillation model
distill_model = DistillModel(teacher_model, N_CLASSES, TEMPERATURE).to(DEVICE)

# Optimizer for the distillation model
distill_optimizer = optim.Adam(distill_model.student_model.parameters(), lr=LEARNING_RATE)

# Training loop for distillation
for epoch in range(N_EPOCHS):
    distill_model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        distill_optimizer.zero_grad()
        student_logits, teacher_logits = distill_model(inputs)
        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
        loss.backward()
        distill_optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Distillation Training')

# Test the distilled model
distill_model.student_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = distill_model.student_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
distill_accuracy = 100 * correct / total
print(f'Accuracy of the distilled model on the CIFAR-10 test images: {distill_accuracy:.2f} %')

# Test the distilled model against FGSM attack
distill_fgsm_accuracy = test_fgsm(distill_model.student_model, test_loader, EPSILON, criterion, DEVICE)


# ART attacks
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=N_CLASSES,
)

# Define the attacks
fgm_attack = FastGradientMethod(estimator=classifier, eps=0.1)
pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=0.3, eps_step=0.1, max_iter=40)
bim_attack = BasicIterativeMethod(estimator=classifier, eps=0.2, eps_step=0.05, max_iter=30)

# Generate adversarial examples
def generate_adversarial_examples(input_batch):
    fgm_adv_examples = fgm_attack.generate(x=input_batch)
    pgd_adv_examples = pgd_attack.generate(x=input_batch)
    bim_adv_examples = bim_attack.generate(x=input_batch)
    return [fgm_adv_examples, pgd_adv_examples, bim_adv_examples]

# Weighted ensemble attack
def weighted_ensemble_attack(attacks, input_batch, weights=None):
    num_attacks = len(attacks)
    if weights is None:
        weights = [1 / num_attacks] * num_attacks  # Equal weights if not provided

    combined_attack = np.zeros_like(input_batch)
    for i in range(num_attacks):
        combined_attack += attacks[i] * weights[i]

    return combined_attack

# Test the model against combined attack
def test_combined_attack(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        attacks = generate_adversarial_examples(images.cpu().numpy())
        combined_adv_images = weighted_ensemble_attack(attacks, images.cpu().numpy())
        combined_adv_images = torch.tensor(combined_adv_images).to(device)
        outputs = model(combined_adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on combined adversarial test images: {accuracy:.2f} %')
    return accuracy

# Test the original model against combined attack
test_combined_attack(model, test_loader, DEVICE)

# Test the distilled model against combined attack
test_combined_attack(distill_model.student_model, test_loader, DEVICE)

