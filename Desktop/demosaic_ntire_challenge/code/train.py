count_high=0
for filename in os.listdir("/kaggle/input/training-set/high"):
    if filename.endswith(".png"):
        count_high+=1
    else:
        print("Error in", filename)
        count_high+=1
print(count_high)

count_low=0
for filename in os.listdir("/kaggle/input/training-set/low"):
    if filename.endswith(".png"):
        count_low+=1
    else:
        print("Error in", filename)
        count_low+=1
print(count_low)

in_channels=3
model=final_imagelab_model(in_channels=in_channels)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model.to(device)

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset,random_split

class PairedImageDataset(Dataset):
    def __init__(self, high_dir, low_dir, transform=None):
        self.high_dir = high_dir
        self.low_dir = low_dir
        self.transform = transform
        self.high_images = sorted(os.listdir(high_dir))
        self.low_images = sorted(os.listdir(low_dir))

    def __len__(self):
        return len(self.high_images)

    def __getitem__(self, idx):
        high_img_path = os.path.join(self.high_dir, self.high_images[idx])
        low_img_path = os.path.join(self.low_dir, self.low_images[idx])
        
        high_image = Image.open(high_img_path).convert("RGB")
        low_image = Image.open(low_img_path).convert("RGB")
        
        if self.transform:
            high_image = self.transform(high_image)
            low_image = self.transform(low_image)
        high_image=high_image.unsqueeze(0)
        low_image=low_image.unsqueeze(0)
        
        return low_image, high_image

    
transform = transforms.Compose([transforms.ToTensor()])
high_dir = "/kaggle/input/dataset00/augmented_Train/augmented/high/"
low_dir = '/kaggle/input/dataset00/augmented_Train/augmented/low/'
dataset = PairedImageDataset(high_dir=high_dir, low_dir=low_dir, transform=transform)
train_size = int(0.8*len(dataset))
val_size = len(dataset)-train_size
train_dataset,val_dataset =random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

criterion=CombinedLoss()
epochs=3
optimizer = optim.Adam(model.parameters(), lr=0.00001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def calculate_psnr(outputs, targets, max_pixel_value=1.0):
    mse = F.mse_loss(outputs, targets)
    psnr = 10 * torch.log10(max_pixel_value**2 / mse)
    return psnr.item()


for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    epoch_loss = 0
    epoch_psnr = 0

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        inputs = inputs.squeeze(1)
        targets = targets.squeeze(1)
        
        outputs = model(inputs)
        outputs=torch.clamp(outputs,0,1)

        loss = criterion(outputs,targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        psnr = calculate_psnr(outputs, targets)
        epoch_psnr += psnr
    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_psnr = epoch_psnr / len(train_loader)
    
    # Validation
    
    model.eval()
    val_loss = 0
    val_psnr = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            inputs=inputs.squeeze(1)
            targets=targets.squeeze(1)
            
            outputs = model(inputs)
            outputs = torch.clamp(outputs,0,1)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            psnr = calculate_psnr(outputs, targets)
            val_psnr += psnr
            
    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss}, Average PSNR: {avg_epoch_psnr:.2f} dB")
    print(f"Average Validation Loss: {avg_val_loss}, Average Validation PSNR: {avg_val_psnr:.2f} dB")
    torch.cuda.empty_cache()