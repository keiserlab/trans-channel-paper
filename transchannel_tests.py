"""
Unit tests for correctness of certain key methods and data structures
The global variable "sample_size" is instantiated below to indicate the number of image examples to test for some of the more computationally and time-intensive methods
"""
from transchannel import *
import unittest

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
        actual = torch.tensor(np.array([[1,1], [0,0]])) #-> 1,1, -1 -1
        predicted = torch.tensor(np.array([[0,0],[1,1]]))# -> -1 -1 -1 1 1 1
        self.assertEqual(calculateMSE(actual, predicted), np.sqrt(16)) #  sqrt(4 + 4 + 4 + 4)
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
        test the getMetrics() function's correctness for TPR, TNR, PPV, NPV, FNR, FPR
        """
        ## case with nonzero true positives, true negatives, and false negatives
        actual = torch.FloatTensor(np.array([[[.99, .99], [.98, .98]]]))
        predicted = torch.FloatTensor(np.array([[[1.1, 1.0],[1.0, 1.0]]]))
        self.assertEqual(getMetrics(predicted, actual, lab_thresh=.97, pred_thresh=.97), (0.5, 1.0, 1.0, 0.6666666666666666, 0.5, 0.0))  #TPR, TNR, PPV, NPV, FNR, FPR
        ## all true negatives case, no positives
        actual = torch.FloatTensor(np.array([[[1.0, 255], [255, 255]]]))
        predicted = torch.FloatTensor(np.array([[[1.0, 255],[255, 255]]]))
        metrics = getMetrics(predicted, actual, lab_thresh=.97, pred_thresh=.97) 
        self.assertTrue(math.isnan(metrics[0]))
        self.assertEqual(metrics[1], 1.0)
        self.assertTrue(math.isnan(metrics[2]))
        self.assertEqual(metrics[3], 1.0)
        self.assertTrue(math.isnan(metrics[4]))
        self.assertEqual(metrics[5], 0)
        ## all true positives case, no negatives
        metrics = getMetrics(predicted, actual, lab_thresh=-100, pred_thresh=-100) 
        self.assertEqual(metrics[0], 1.0)
        self.assertTrue(math.isnan(metrics[1]))
        self.assertEqual(metrics[2], 1.0)
        self.assertTrue(math.isnan(metrics[3]))
        self.assertEqual(metrics[4], 0)
        self.assertTrue(math.isnan(metrics[5]))

class MainFunctionsTests(unittest.TestCase):
    """
    test for the more detailed and sophisticated functions that require loading the ML model
    """
    csv_name = "raw_dataset_1_thru_6_full_images_gpu2.csv"
    meanSTDStats = "raw_dataset_1_thru_6_stats.npy"
    minMaxStats = "raw_1_thru_6_min_max.npy" #stats for min max values 
    stats = np.load(meanSTDStats)
    inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
    stats = np.load(minMaxStats)
    inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
    dataset = ImageDataset(csv_name, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax)
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

    def testGetROC(self):
        """
        test the method getROC()
        """
        ML_x, ML_y, null_YFP_x, null_YFP_y, null_DAPI_x, null_DAPI_y = getROC(lab_thresh=1.0, sample_size=sample_size, model=self.model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=self.validation_generator, device=self.device)
        # print(ML_x, ML_y, null_YFP_x, null_YFP_y, null_DAPI_x, null_DAPI_y)
        ## make sure return is in expected, sorted form 
        self.assertTrue(ML_x == sorted(ML_x, reverse=True))
        self.assertTrue(null_YFP_x == sorted(null_YFP_x, reverse=True))
        self.assertTrue(null_DAPI_x == sorted(null_DAPI_x, reverse=True))
        self.assertTrue(ML_y == sorted(ML_y, reverse=True))
        self.assertTrue(null_YFP_y == sorted(null_YFP_y, reverse=True))
        self.assertTrue(null_DAPI_y == sorted(null_DAPI_y, reverse=True))
        ## make sure return is well shaped and of same lengths 
        self.assertTrue(len(ML_x) == len(ML_y) == len(null_YFP_x) == len(null_YFP_y) == len(null_DAPI_x) == len(null_YFP_y))
        ## make sure in right data range [0, 1] inclusive 
        self.assertTrue(all(0.0 <= element <= 1.0 for element in ML_x))
        self.assertTrue(all(0.0 <= element <= 1.0 for element in ML_y))
        self.assertTrue(all(0.0 <= element <= 1.0 for element in null_YFP_x))
        self.assertTrue(all(0.0 <= element <= 1.0 for element in null_YFP_y))
        self.assertTrue(all(0.0 <= element <= 1.0 for element in null_DAPI_x))
        self.assertTrue(all(0.0 <= element <= 1.0 for element in null_DAPI_y)) 
        ## make sure  null DAPI performance < null YFP performance < ML performance 
        ML_auc = -1 * np.trapz(ML_y, ML_x)
        null_auc = -1 * np.trapz(null_YFP_y, null_YFP_x)
        null_DAPI_auc = -1 * np.trapz(null_DAPI_y, null_DAPI_x)
        self.assertTrue(null_DAPI_auc < null_auc < ML_auc)

class DatasetTests(unittest.TestCase):
    """
    test the different dataset objects
    """
    def testImageDataset(self):
        """
        test the Dataset object used for the tauopathy training set
        """
        csv_name = "raw_dataset_1_thru_6_full_images_gpu2.csv"
        meanSTDStats = "raw_dataset_1_thru_6_stats.npy"
        minMaxStats = "raw_1_thru_6_min_max.npy" #stats for min max values 
        stats = np.load(meanSTDStats)
        inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
        stats = np.load(minMaxStats)
        inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
        dataset = ImageDataset(csv_name, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax)
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
        csv_name = "raw_dataset_1_thru_6_full_images_gpu2.csv"
        meanSTDStats = "raw_dataset_1_thru_6_stats.npy"
        minMaxStats = "raw_1_thru_6_min_max.npy" #stats for min max values 
        stats = np.load(meanSTDStats)
        inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
        stats = np.load(minMaxStats)
        inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
        dataset = DAPIDataset(csv_name, DAPIMin, DAPIMax, labelMin, labelMax)
        generator = data.DataLoader(dataset, sampler = SubsetRandomSampler(list(range(0, len(dataset)))))
        i = 0
        ## iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            self.assertTrue("DAPI" in names[0])
            ## make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ## make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 2048, 2048))
            self.assertEqual(tuple(local_labels.shape), (1, 2048, 2048))
            i += 1
            if i > sample_size:
                break

    def testOsteosarcomaDataset(self):
        """
        test the Dataset used for the osteosarcoma dataset
        """
        csvName = "/home/wongd/phenoScreen/datasets/cyclin_dataset.csv"
        dataset = OsteosarcomaDataset(csvName)
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
        csvName = "/home/wongd/phenoScreen/datasets/cyclin_dataset.csv"
        dataset = OsteosarcomaAblationDataset(csvName, thresh_percent=1.0) #full ablation dataset - all channel 0 input pixels should be fully ablated and set to 0 value
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

sample_size = 10
unittest.main()
