from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.folder import *
import os

def find_classes_open(dir):
    
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    
    if 'openset' in classes:

        classes.remove('openset')
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        classes.append('openset')
        class_to_idx['openset'] = -1
    else:
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
    
    return classes, class_to_idx

# ImageFolder with path and open class labeled as -1

class ImageFolderWithPathAndOpen(datasets.ImageFolder):
    
    def __init__(self, root, transform=None):

        super(ImageFolderWithPathAndOpen, self).__init__(root=root, 
                                                  transform=transform)

        classes, class_to_idx = find_classes_open(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        
    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
        
# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets
def load_data(data_root, dataset, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):
    
    # assert dataset in ['train', 'test', 'val', 'train_plain', 'test_open']
    
    if dataset == 'test' and test_open:
        print('Using open test set.')
        dataset = 'test_open'

    dataset_dirs = os.path.join(data_root, (dataset if dataset != 'train_plain' else 'train')) 

    # consider imagenet tailness experiments.
    if not os.path.exists(dataset_dirs) and dataset == 'test':
        dataset_dirs = '/home/public/dataset/imagenet_LT/test'
    elif not os.path.exists(dataset_dirs) and dataset == 'test_open':
        dataset_dirs = '/home/public/dataset/imagenet_LT/test_open'
    
    print('Loading data from %s' % (dataset_dirs))
    
    if dataset not in ['train', 'val']:
        transform = data_transforms['test']
    else:
        transform = data_transforms[dataset]

    print('Use data transformation:', transform)
    
    set_ = ImageFolderWithPathAndOpen(root=dataset_dirs,
                                      transform=transform)
        
    if sampler_dic and dataset == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return (DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers),
               len(set_))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return (DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers), 
               len(set_))


        
    
    
