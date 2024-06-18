from torchvision import transforms

def augment_and_save_images(input_folder,output_folder,num_augmentations=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_transforms=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    for imgname in os.listdir(input_folder):
        img_path=os.path.join(input_folder, imgname)
        image=Image.open(img_path).convert("RGB")
 
        for i in range(num_augmentations):
            augimage=data_transforms(image)
            augimg_name=f"{os.path.splitext(img_name)[0]}_aug_{i+1}{os.path.splitext(img_name)[1]}"
            augimg_path=os.path.join(output_folder, augimg_name)
            augimage.save(augmented_img_path)

input_folder='/kaggle/input/training-set/high'
output_folder='/kaggle/working/augmented_high'
augment_and_save_images(input_folder, output_folder, num_augmentations=3)