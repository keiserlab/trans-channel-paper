"""
Script to run and call transchannel.py,
This script can be run with the command "python transchannel_runner.py {FOLD NUMBER}". 
We used a 3-fold cross-validation scheme over an extended dataset for the supplemental analysis (specify this by setting FOLD NUMBER to either 1,2, or 3).
For the main paper analyses, we performed a 70% train, 30% test split presented. To specify this setup, give an integer argument that is not in [1,2,3] for FOLD NUMBER. 
consists of two sections
1) global variables and parameters 
2) method calls to run specific parts of the code

"""
#============================================================================
#============================================================================
## VARIABLES AND PARAMETERS
#============================================================================
#============================================================================
from transchannel import *

hostname = socket.gethostname() 
if torch.cuda.device_count() > 1:
  print("# available GPUs", torch.cuda.device_count(), "GPUs!")
use_cuda = torch.cuda.is_available()
##fold of the 3-fold cross-validation to use (1,2, or 3), else any other integer will specify no cross-validation and use a random 70% training, 30% test split
fold = int(sys.argv[1]) 
if fold in [1,2,3]:
    cross_val = True
else:
    cross_val = False
plotName = "MODEL_NAME".format(fold) #name used to save model 
if "keiserlab" in hostname: ##if on keiser lab server, else butte lab server 
    if cross_val:
        csv_name = "/srv/home/dwong/AIInCell/datasets/butte_server_raw_plates_1_thru_16.csv"
    else:
        csv_name = "/srv/home/dwong/AIInCell/datasets/raw_dataset_1_thru_6_full_images_gpu2.csv" #plates 1 through 6 on gpu2's fast drive, used to train model for historical validation 
    meanSTDStats = "/srv/home/dwong/AIInCell/models/raw_dataset_1_thru_6_stats.npy"
    minMaxStats = "/srv/home/dwong/AIInCell/models/raw_1_thru_6_min_max.npy" #stats for min max values 
    train_params = {'batch_size': 1, 'num_workers': 6} #batch size 32 for tiles, 4 seems ok for full image
    test_params = {'batch_size': 1, 'num_workers': 6} 
else:
    if cross_val:
        csv_name = "butte_server_raw_plates_1_thru_16.csv"
    else:
        csv_name = "raw_dataset_1_thru_6_full_images_gpu2.csv"
    meanSTDStats = "raw_dataset_1_thru_6_stats.npy"
    minMaxStats = "raw_1_thru_6_min_max.npy" #stats for min max values 
    train_params = {'batch_size': 1, 'num_workers': 2} #batch size 32 for tiles, 4 seems ok for full image
    test_params = {'batch_size': 1, 'num_workers': 2} 
max_epochs = 20
learning_rate = .001
continue_training = False ##if we want to train from a pre-trained model
if continue_training:
    load_training_name = "LOAD_MODEL_NAME.pt" #model to use if we're training from a pre-trained model
gpu_list = [0,1] ##gpu ids to use
normalize = "scale" #scale for scaling values 0 to 255, or "unit" for subtracting mean and dividing by std 
lossfn = pearsonCorrLoss 
architecture = "unet mod"
device = torch.device("cuda:" + str(gpu_list[0]) if use_cuda else "cpu")
##Initialize stats, model, loss function, and optimizer 
stats = np.load(meanSTDStats)
inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
stats = np.load(minMaxStats)
inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
model = Unet_mod(inputChannels=2)
if len(gpu_list) > 1:
    model = nn.DataParallel(model, device_ids=gpu_list).cuda()
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
## Random seed for data split 
dataset = ImageDataset(csv_name, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(.3 * dataset_size)) #30% test 
seed = 42
np.random.seed(seed)
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
    test_sampler = SubsetRandomSampler(test_indices) #randomize indices 
else: #if we're specifying a 70% train, 30% test split 
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices) #radom indices 
training_generator = data.DataLoader(dataset, sampler=train_sampler, **train_params)
validation_generator = data.DataLoader(dataset,sampler=test_sampler, **test_params)


#============================================================================
#============================================================================
## BRIEF TESTS
#============================================================================
#============================================================================
##make sure train and test are separate and there's zero overlap


#============================================================================
#============================================================================
## METHOD CALLS
#============================================================================
#============================================================================
# train(continue_training=False, model=model, max_epochs=max_epochs, training_generator=training_generator, validation_generator=validation_generator, lossfn=lossfn, optimizer=optimizer, plotName="plotName",device=device)
test(sample_size=100, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=validation_generator, lossfn=lossfn,device=device)
# testOnSeparatePlates(sample_size=1000, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=validation_generator, lossfn=lossfn, device=device)
# getMSE(loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", model=model, validation_generator=validation_generator, device=device)
# getNull(validation_generator=validation_generator,device=device)
# getROC(lab_thresh=1.0, sample_size=1000000, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=validation_generator, device=device)
# ablationTestTau(sample_size=1000000, validation_generator=validation_generator, ablate_DAPI_only=False, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt",device=device)

