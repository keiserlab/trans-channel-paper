"""
Unit tests for certain key methods and data structures
The global variable "sample_size" is instantiated below to indicate the number of image examples to test for some of the more computationally and time-intensive methods
"""
from transchannel import *
import unittest

hostname = socket.gethostname() 

class HelperFunctionsTests(unittest.TestCase):
    """
    tests for the helper functions
    """
    def testCalculateMSE(self):
        """
        test the MSE function
        """
        ## matching case
        actual = torch.tensor(np.array([[1,0,1], [1,1,1], [0,0,0]]))
        predicted = torch.tensor(np.array([[1,0,1], [1,1,1], [0,0,0]]))
        self.assertEqual(calculateMSE(actual, predicted),0)
        ## non-matching case with error
        actual = torch.tensor(np.array([[1,1], [0,0]])) 
        predicted = torch.tensor(np.array([[0,0],[1,1]]))
        self.assertEqual(calculateMSE(actual, predicted), np.sqrt(16)) 
        self.assertNotEqual(4,5)

    def testPearsonCorrLoss(self):
        """
        test the accuracy of the custom negative pearson loss function
        """
        ## perfect prediction
        actual = torch.FloatTensor(np.array([[.9,.9], [1,1]]))
        predicted = torch.FloatTensor(np.array([[.9,.9],[1,1]]))
        self.assertEqual(pearsonCorrLoss(actual, predicted), -1)
        ## imperfect prediction
        actual = torch.FloatTensor(np.array([[.9,.8], [1,1]])).type(torch.float32) 
        predicted = torch.FloatTensor(np.array([[.01,.9],[.01,.01]])).type(torch.float32)
        self.assertTrue(abs(pearsonCorrLoss(actual, predicted).item() - 0.87038) <= .001)

    def testGetMetrics(self):
        """
        test the getMetrics() function's correctness
        """
        ## case with nonzero true positives, true negatives, and false negatives
        actual = torch.FloatTensor(np.array([[[1.1, 1.1], [0, .99]]]))
        predicted = torch.FloatTensor(np.array([[[1.05, .99],[.99, 1.1]]]))
        self.assertEqual(getMetrics(predicted, actual, lab_thresh=1, pred_thresh=1), (1, 1, 1, 1)) # true_positive, false_positive, true_negative, false_negative
        ## all true negatives case, no positives
        actual = torch.FloatTensor(np.array([[[1.0, 1.9], [1.9, 1.9]]]))
        predicted = torch.FloatTensor(np.array([[[1.0, 1.9],[1.9, 1.9]]]))
        metrics = getMetrics(predicted, actual, lab_thresh=2.0, pred_thresh=2.0)
        self.assertEqual(metrics, (0, 0, 4, 0))
        ## all true positives case, no negatives
        metrics = getMetrics(predicted, actual, lab_thresh=-100, pred_thresh=-100) 
        self.assertEqual(metrics, (4, 0, 0, 0))

class MainFunctionsTests(unittest.TestCase):
    """
    test for the more detailed functions that require loading the ML model
    """
    csv_name = "csvs/raw_dataset_1_thru_6_full_images_gpu2.csv"
    meanSTDStats = "stats/raw_dataset_1_thru_6_stats.npy"
    minMaxStats = "stats/raw_1_thru_6_min_max.npy" #stats for min max values  
    if "keiser" in hostname:
        DATA_DIR = "/srv/nas/mk3/users/dwong/" #where the raw images are located
    else:
        DATA_DIR = "/data1/wongd/"
    stats = np.load(meanSTDStats)
    inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
    stats = np.load(minMaxStats)
    inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
    dataset = ImageDataset(csv_name, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax, DATA_DIR)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(.3 * dataset_size)) #30% test 
    seed = 42
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    test_sampler = SubsetRandomSampler(test_indices) #radom indices 
    test_params = {'batch_size': 1, 'num_workers': 3} 
    validation_generator = data.DataLoader(dataset,sampler=test_sampler, **test_params)
    model = Unet_mod(inputChannels=2)
    model = nn.DataParallel(model, device_ids=[0,1]).cuda()
    device = torch.device("cuda:0")
    model = model.to(device)

    def testModelEvaluation(self):
        """
        test the method test()
        """
        test_results = test(sample_size=sample_size, model=self.model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=self.validation_generator, lossfn=pearsonCorrLoss, device=self.device) #iterate over a few random test images 
        ## ml pearson model > null 
        self.assertTrue(test_results[0][0] > test_results[1][0]) 
        ## mse model < null 
        self.assertTrue(test_results[2][0] < test_results[3][0]) 
        ##make sure bounds are correct for both pearson and MSE
        self.assertTrue(0 <= test_results[0][0] <= 1)
        self.assertTrue(0 <= test_results[2][0])
        
    def testTrainTestSplit(self):
        """
        make sure train and test indices do not overlap
        """
        self.assertEqual(len(set(self.train_indices).intersection(self.test_indices)), 0) 

    def testROCAndPRC(self):
        """
        test the performance curve results for validity 
        """
        mapp = pickle.load(open("pickles/mapp_fold_-1.pk", "rb"))
        null_mapp = pickle.load(open("pickles/null_YFP_mapp_fold_-1.pk", "rb"))
        null_DAPI_mapp = pickle.load(open("pickles/null_DAPI_mapp_fold_-1.pk", "rb"))
        for m in [mapp, null_mapp, null_DAPI_mapp]:
            for key in m: 
                TPs = sum([x[0] for x in m[key]])
                FPs = sum([x[1] for x in m[key]])
                TNs = sum([x[2] for x in m[key]])
                FNs = sum([x[3] for x in m[key]])
                positives = TPs + FNs 
                total = TPs + FPs + TNs + FNs 
                num_images = len(m[key])
                positive_prevalence = positives / float(total)
                self.assertTrue(num_images * 2048 * 2048 == total) ##make sure we are accounting for each pixel 
                self.assertTrue(num_images == 17280) ##make sure every image in test set is accounted for 

class DatasetTests(unittest.TestCase):
    """
    test the different dataset objects
    """
    def testImageDataset(self):
        """
        test the Dataset object used for the tauopathy training set
        """
        csv_name = "csvs/raw_dataset_1_thru_6_full_images_gpu2.csv"
        meanSTDStats = "stats/raw_dataset_1_thru_6_stats.npy"
        minMaxStats = "stats/raw_1_thru_6_min_max.npy" #stats for min max values
        if "keiser" in hostname:
            DATA_DIR = "/srv/nas/mk3/users/dwong/" #where the raw images are located
        else:
            DATA_DIR = "/data1/wongd/"
        stats = np.load(meanSTDStats)
        inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
        stats = np.load(minMaxStats)
        inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
        dataset = ImageDataset(csv_name, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax, DATA_DIR)
        generator = data.DataLoader(dataset, sampler = SubsetRandomSampler(list(range(0, len(dataset)))))
        i = 0
        ## iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            self.assertTrue("FITC" in names[0])
            ## make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ## make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 2, 2048, 2048))
            self.assertEqual(tuple(local_labels.shape), (1, 2048, 2048))
            i += 1
            if i > sample_size:
                break

    def testDAPIDataset(self):
        """
        test the DAPI Dataset object used for to train DAPI -> AT8 (supplement)
        """
        csv_name = "csvs/raw_dataset_1_thru_6_full_images_gpu2.csv"
        meanSTDStats = "stats/raw_dataset_1_thru_6_stats.npy"
        minMaxStats = "stats/raw_1_thru_6_min_max.npy" #stats for min max values 
        if "keiser" in hostname:
            DATA_DIR = "/srv/nas/mk3/users/dwong/" #where the raw images are located
        else:
            DATA_DIR = "/data1/wongd/"
        stats = np.load(meanSTDStats)
        inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
        stats = np.load(minMaxStats)
        inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
        dataset = DAPIDataset(csv_name, DATA_DIR)
        generator = data.DataLoader(dataset, sampler = SubsetRandomSampler(list(range(0, len(dataset)))))
        i = 0
        ## iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            self.assertTrue("DAPI" in names[0])
            ## make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ## make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 1, 2048, 2048))
            self.assertEqual(tuple(local_labels.shape), (1, 2048, 2048))
            i += 1
            if i > sample_size:
                break

    def testOsteosarcomaDataset(self):
        """
        test the Dataset used for the osteosarcoma dataset
        """
        if "keiser" in hostname:
            DATA_DIR = "/srv/nas/mk1/users/dwong/tifs/" #where the raw images are located
        else:
            DATA_DIR = "/data1/wongd/tifs/"
        csvName = "csvs/cyclin_dataset.csv"
        dataset = OsteosarcomaDataset(csvName, DATA_DIR)
        generator = data.DataLoader(dataset, sampler = SubsetRandomSampler(list(range(0, len(dataset)))))
        i = 0
        ## iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            ## make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ## make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 1, 1104, 1104))
            self.assertEqual(tuple(local_labels.shape), (1, 1104, 1104))
            i += 1
            if i > sample_size:
                break

    def testOsteosarcomaAblationDataset(self):
        """
        test the Dataset used for training the ablated osteosarcoma model
        """
        csvName = "csvs/cyclin_dataset.csv"
        if "keiser" in hostname:
            DATA_DIR = "/srv/nas/mk1/users/dwong/tifs/" #where the raw images are located
        else:
            DATA_DIR = "/data1/wongd/tifs/"
        dataset = OsteosarcomaAblationDataset(csvName, DATA_DIR, thresh_percent=1.0) #full ablation dataset - all channel 0 input pixels should be fully ablated and set to 0 value
        generator = data.DataLoader(dataset, sampler = SubsetRandomSampler(list(range(0, len(dataset)))))
        i = 0
        ## iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            ## make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ## make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 1, 1104, 1104))
            self.assertEqual(tuple(local_labels.shape), (1, 1104, 1104))
            ## make sure all of input is ablated
            self.assertEqual(np.count_nonzero(local_batch.cpu().numpy()), 0)
            i += 1
            if i > sample_size:
                break

class ArchitectureTests(unittest.TestCase):
    """
    test to make sure number of parameters matches with that computed by hand
    """
    def testSize(self):
        device = torch.device("cuda:0")
        model = Unet_mod(inputChannels=2)
        model = nn.DataParallel(model, device_ids=[0,1]).cuda()
        model = model.to(device)
        checkpoint = torch.load("models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        total_params = 0
        for name, parameter in model.named_parameters():
            total_params += parameter.numel()
        self.assertEqual(total_params -(512*512*2*2) - 512,3814401) ##left in an unused param, historic artifact


sample_size = 1000000000
unittest.main()

