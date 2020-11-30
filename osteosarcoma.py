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
if "keiser" in hostname:
    DATA_DIR = "/srv/nas/mk1/users/dwong/tifs/"
else:
    DATA_DIR = "/data1/wongd/tifs/"
short = False #param for truncating training and testing process for quick training dev
img_dim = 1104
train_params = {'batch_size': 1, 'num_workers': 3}
test_params = {'batch_size': 1, 'num_workers': 3}
plotName = "MODEL_NAME" #name of model to save
csvName = "csvs/cyclin_dataset.csv"
gpu_list = [1] 
max_epochs = 5
learning_rate = .001
continue_training = False
if continue_training:
    load_training_name = "LOAD_MODEL_NAME.pt"
lossfn = pearsonCorrLoss
architecture = "unet mod"
fold = int(sys.argv[1]) #specify cross validation fold to use in [1,2,3], else any other integer will do a 70% train, 30% test split
task = sys.argv[2] #either "raw" or "ablation" to determine how to process the images, ablated = 95% intensity ablation
if fold in [1,2,3]:
    cross_val = True
else:
    cross_val = False
device = torch.device("cuda:" + str(gpu_list[0]))
## Generators
if task == "raw":
    dataset = OsteosarcomaDataset(csvName, DATA_DIR)
if task == "ablation":
    dataset = OsteosarcomaAblationDataset(csvName, DATA_DIR, .95)
print("task: ", task)
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
shutil.rmtree("outputs/")
os.mkdir("outputs/")

#=======================================================================================
#=======================================================================================
#METHOD CALLS 
#=======================================================================================
#=======================================================================================

train(continue_training=False, model=model, max_epochs=max_epochs, training_generator=training_generator, validation_generator=validation_generator, lossfn=lossfn, optimizer=optimizer, plotName="null",device=device)
if task == "raw":
    ml_model_perf, null_model_perf, ml_model_mse_perf, null_model_mse_perf = test(sample_size=10000000, model=model, loadName="models/d0_to_d1_cyclin_only_dataset_fold{}.pt".format(fold), validation_generator=validation_generator, lossfn=lossfn,device=device)
    pickle.dump(ml_model_perf, open("pickles/osteo_ml_model_perf_fold_{}.pkl".format(fold), "wb"))
    pickle.dump(null_model_perf, open("pickles/osteo_null_model_perf_fold_{}.pkl".format(fold), "wb"))
    pickle.dump(ml_model_mse_perf, open("pickles/osteo_ml_model_mse_perf_fold_{}.pkl".format(fold), "wb"))
    pickle.dump(null_model_mse_perf, open("pickles/osteo_null_model_mse_perf_fold_{}.pkl".format(fold), "wb"))

if task == "ablation":
    ml_model_perf, null_model_perf, ml_model_mse_perf, null_model_mse_perf = test(sample_size=10000000, model=model, loadName="models/d0_to_d1_ablation_cyclin_only_dataset_fold{}_continue_training.pt".format(fold), validation_generator=validation_generator, lossfn=lossfn,device=device)
    pickle.dump(ml_model_perf, open("pickles/osteo_ablated_ml_model_perf_fold_{}.pkl".format(fold), "wb"))
    pickle.dump(null_model_perf, open("pickles/osteo_ablated_null_model_perf_fold_{}.pkl".format(fold), "wb"))
    pickle.dump(ml_model_mse_perf, open("pickles/osteo_ablated_ml_model_mse_perf_fold_{}.pkl".format(fold), "wb"))
    pickle.dump(null_model_mse_perf, open("pickles/osteo_ablated_null_model_mse_perf_fold_{}.pkl".format(fold), "wb"))
osteosarcomaAblatedAndNonAblated(sample_size=10000000, validation_generator=validation_generator, model=model, fold=fold, device=device)




