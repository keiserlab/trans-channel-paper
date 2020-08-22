"""
Script to train and evaluate models on the independent dataset of the genome-wide functional screen in osteosarcoma cells 

This script is divided into three parts:
1) class and method definitions
2) global variables
3) method calls to run specific parts of the code

Any questions should be directed to daniel.wong2@ucsf.edu. Thank you!

"""
from transchannel import *

#=======================================================================================
#=======================================================================================
#GLOBAL VARIABLES
#=======================================================================================
#=======================================================================================

hostname = socket.gethostname() 
if "keiserlab.org" not in hostname:
    prefix = "/data1/wongd/"
else:
    prefix = "/srv/nas/mk1/users/dwong/"
task = "transchannel" #for training and testing the learner on raw, unablated images
# task = "ablation" ##for training and testing the learner on ablated images at the 95th percentile intensity
short = False #param for truncating training and testing process for quick training dev
img_dim = 1104
train_params = {'batch_size': 1, 'num_workers': 5}
test_params = {'batch_size': 1, 'num_workers': 5}
plotName = "MODEL_NAME" #name of model to save
csvName = "datasets/cyclin_dataset.csv"
gpu_list = [1] 
max_epochs = 5
learning_rate = .001
continue_training = False
if continue_training:
    load_training_name = "LOAD_MODEL_NAME.pt"
lossfn = pearsonCorrLoss
architecture = "unet mod"
fold = int(sys.argv[1]) #specify cross validation fold to use in [1,2,3], else any other integer will do a 70% train, 30% test split
if fold in [1,2,3]:
    cross_val = True
else:
    cross_val = False
device = torch.device("cuda:" + str(gpu_list[0]))
## Generators
if task == "transchannel":
    dataset = OsteosarcomaDataset(csvName)
if task == "ablation":
    dataset = OsteosarcomaAblationDataset(csvName, .95)
## Random seed for data split 
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(.3 * dataset_size)) #30% test 
np.random.seed(42)
np.random.shuffle(indices)
if cross_val:
    split1 = int(np.floor(.3333 * dataset_size)) #1/3 test 
    split2 = int(np.floor(.6666667 * dataset_size)) #2/3 train 
    if fold == 1:
        train_indices, test_indices = indices[0:split2], indices[split2:] #[train, train, test]
    if fold == 2:
        train_indices, test_indices = indices[split1:], indices[0:split1] #[test, train, train]
    if fold == 3:
        train_indices, test_indices = indices[0:split1] + indices[split2:], indices[split1:split2] #[train, test, train]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices) 
else:
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices) 
training_generator = data.DataLoader(dataset, sampler=train_sampler, **train_params)
validation_generator = data.DataLoader(dataset,sampler=test_sampler, **test_params)
if architecture == "unet mod":
    model = Unet_mod_osteo(inputChannels=1)
if len(gpu_list) > 1:
    model = nn.DataParallel(model, device_ids=gpu_list).cuda()
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
if continue_training:
    checkpoint = torch.load(load_training_name,  map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()


#=======================================================================================
#=======================================================================================
#METHOD CALLS 
#=======================================================================================
#=======================================================================================

train(continue_training=False, model=model, max_epochs=max_epochs, training_generator=training_generator, validation_generator=validation_generator, lossfn=lossfn, optimizer=optimizer, plotName="plotName",device=device)
test(sample_size=1000000, model=model, loadName="models/d0_to_d1_cyclin_only_dataset_fold{}.pt".format(fold), validation_generator=validation_generator, lossfn=lossfn,device=device)
getMSE(loadName="models/d0_to_d1_cyclin_only_dataset_fold{}.pt".format(fold), model=model, validation_generator=validation_generator, device=device)
ablationTestOsteosarcoma(sample_size=1000000, validation_generator=validation_generator, model=model, loadName="models/d0_to_d1_cyclin_only_dataset_fold{}.pt".format(fold),device=device)




