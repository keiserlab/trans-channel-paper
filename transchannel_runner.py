"""
Script for runing code pertinent to the tauopathy study,
This script can be run with the command "python transchannel_runner.py {FOLD NUMBER}". 
If you wish to perform a 3-fold cross-validation scheme instead of the 70% train 30% test split used in this study, you can specify FOLD NUMBER = 1,2, or 3 
To specify the main analyses presented in this study, give an integer argument that is not in [1,2,3] for FOLD NUMBER. 
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
if "keiser" in hostname:
    DATA_DIR = "/srv/nas/mk3/users/dwong/" #where the raw images are located
else:
    DATA_DIR = "/data1/wongd/"
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
csv_name = "csvs/raw_dataset_1_thru_6_full_images_gpu2.csv" 
meanSTDStats = "stats/raw_dataset_1_thru_6_stats.npy"
minMaxStats = "stats/raw_1_thru_6_min_max.npy" #stats for min max values 
train_params = {'batch_size': 1, 'num_workers': 6} 
test_params = {'batch_size': 1, 'num_workers': 6} 
max_epochs = 30
learning_rate = .001
continue_training = False ##if we want to continue training from a pre-trained model
if continue_training:
    load_training_name = "LOAD_MODEL_NAME.pt" #model to use if we're training from a pre-trained model
gpu_list = [0,1] ##gpu ids to use
print("GPUs to use: ", gpu_list)
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
dataset = ImageDataset(csv_name, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax, DATA_DIR)
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
full_data_sampler = SubsetRandomSampler(indices) 
full_data_generator = data.DataLoader(dataset, sampler=full_data_sampler, **train_params)

#============================================================================
#============================================================================
## METHOD CALLS
#============================================================================
#============================================================================

##clear outputs directory first
shutil.rmtree("outputs/")
os.mkdir("outputs/")

train(continue_training=False, model=model, max_epochs=max_epochs, training_generator=training_generator, validation_generator=validation_generator, lossfn=lossfn, optimizer=optimizer, plotName="null",device=device)
ml_model_perf, null_model_perf, ml_model_mse_perf, null_model_mse_perf = test(sample_size=1000000, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=validation_generator, lossfn=lossfn,device=device)
pickle.dump(ml_model_perf, open("pickles/ml_model_perf.pkl", "wb"))
pickle.dump(null_model_perf, open("pickles/null_model_perf.pkl", "wb"))
pickle.dump(ml_model_mse_perf, open("pickles/ml_model_mse_perf.pkl", "wb"))
pickle.dump(null_model_mse_perf, open("pickles/null_model_mse_perf.pkl", "wb"))
ML_x, ML_y, null_YFP_x, null_YFP_y, null_DAPI_x, null_DAPI_y = getROC(lab_thresh=1.0, sample_size=1000000, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=validation_generator, device=device)

##Supplemental analysis
getOverlap(sample_size=1000000, generator=full_data_generator)
testOnSeparatePlates(sample_size=1000000, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=validation_generator, lossfn=lossfn, device=device)
ablationTestTau(sample_size=1000000, validation_generator=validation_generator, ablate_DAPI_only=False, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt",device=device)

##for supplemental DAPI training analysis 
dapi_model = Unet_mod(inputChannels=1)
dapi_model = nn.DataParallel(dapi_model, device_ids=gpu_list).cuda()
dapi_model = dapi_model.to(device)
dataset = DAPIDataset(csv_name, DATA_DIR)
training_generator = data.DataLoader(dataset, sampler=train_sampler, **train_params)
validation_generator = data.DataLoader(dataset,sampler=test_sampler, **test_params)
train(continue_training=False, model=dapi_model, max_epochs=max_epochs, training_generator=training_generator, validation_generator=validation_generator, lossfn=lossfn, optimizer=optimizer, plotName="null",device=device)
ml_model_perf, null_model_perf, ml_model_mse_perf, null_model_mse_perf = test(sample_size=1000000, model=dapi_model, loadName="models/DAPI_to_AT8.pt", validation_generator=validation_generator, lossfn=lossfn,device=device)
pickle.dump(ml_model_perf, open("pickles/single_channel_DAPI_ml_model_perf.pkl", "wb"))
pickle.dump(null_model_perf, open("pickles/single_channel_DAPI_null_model_perf.pkl", "wb"))
pickle.dump(ml_model_mse_perf, open("pickles/single_channel_DAPI_ml_model_mse_perf.pkl", "wb"))
pickle.dump(null_model_mse_perf, open("pickles/single_channel_DAPI_null_model_mse_perf.pkl", "wb"))

##for supplemental YFP only training analysis 
yfp_model = Unet_mod(inputChannels=1)
yfp_model = nn.DataParallel(yfp_model, device_ids=gpu_list).cuda()
yfp_model = yfp_model.to(device)
dataset = YFPDataset(csv_name, DATA_DIR)
training_generator = data.DataLoader(dataset, sampler=train_sampler, **train_params)
validation_generator = data.DataLoader(dataset,sampler=test_sampler, **test_params)
train(continue_training=False, model=yfp_model, max_epochs=max_epochs, training_generator=training_generator, validation_generator=validation_generator, lossfn=lossfn, optimizer=optimizer, plotName="null",device=device)
ml_model_perf, null_model_perf, ml_model_mse_perf, null_model_mse_perf = test(sample_size=1000000, model=yfp_model, loadName="models/YFP_only_to_AT8.pt", validation_generator=validation_generator, lossfn=lossfn,device=device)
pickle.dump(ml_model_perf, open("pickles/single_channel_YFP_ml_model_perf.pkl", "wb"))
pickle.dump(null_model_perf, open("pickles/single_channel_YFP_null_model_perf.pkl", "wb"))
pickle.dump(ml_model_mse_perf, open("pickles/single_channel_YFP_ml_model_mse_perf.pkl", "wb"))
pickle.dump(null_model_mse_perf, open("pickles/single_channel_YFP_null_model_mse_perf.pkl", "wb"))


