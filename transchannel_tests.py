"""
Unit tests for correctness of certain key methods and data structures
"""
from transchannel import *
import unittest

class HelperFunctionsTests(unittest.TestCase):
    def testCalculateMSE(self):
        ##matching case
        actual = torch.tensor(np.array([[1,0,1], [1,1,1], [0,0,0]]))
        predicted = torch.tensor(np.array([[1,0,1], [1,1,1], [0,0,0]]))
        self.assertEqual(calculateMSE(actual, predicted),0)
        ##non-matching case with error
        actual = torch.tensor(np.array([[1,1], [0,0]])) #-> 1,1, -1 -1
        predicted = torch.tensor(np.array([[0,0],[1,1]]))# -> -1 -1 -1 1 1 1
        self.assertEqual(calculateMSE(actual, predicted), np.sqrt(16)) #  sqrt(4 + 4 + 4 + 4)
        self.assertNotEqual(4,5)

    def testPearsonCorrLoss(self):
        actual = torch.FloatTensor(np.array([[.9,.9], [1,1]]))
        predicted = torch.FloatTensor(np.array([[.9,.9],[1,1]]))
        self.assertEqual(pearsonCorrLoss(actual, predicted), -1)
        # actual = torch.FloatTensor(np.array([[.9,.9], [1,1]])).type(torch.float32) 
        # predicted = torch.FloatTensor(np.array([[0.01,0.01],[0.01,0.01]])).type(torch.float32)
        # self.assertEqual(pearsonCorrLoss(actual, predicted), -1)

    def testGetMetrics(self):
        ## case with nonzero true positives, true negatives, and false negatives
        actual = torch.FloatTensor(np.array([[[.99, .99], [.98, .98]]]))
        predicted = torch.FloatTensor(np.array([[[1.1, 1.0],[1.0, 1.0]]]))
        self.assertEqual(getMetrics(predicted, actual, lab_thresh=.97, pred_thresh=.97), (0.5, 1.0, 1.0, 0.6666666666666666, 0.5, 0.0))  #TPR, TNR, PPV, NPV, FNR, FPR
        ##all true negatives case, no positives
        actual = torch.FloatTensor(np.array([[[1.0, 255], [255, 255]]]))
        predicted = torch.FloatTensor(np.array([[[1.0, 255],[255, 255]]]))
        metrics = getMetrics(predicted, actual, lab_thresh=.97, pred_thresh=.97) 
        self.assertTrue(math.isnan(metrics[0]))
        self.assertEqual(metrics[1], 1.0)
        self.assertTrue(math.isnan(metrics[2]))
        self.assertEqual(metrics[3], 1.0)
        self.assertTrue(math.isnan(metrics[4]))
        self.assertEqual(metrics[5], 0)
        ##all true positives case, no negatives
        metrics = getMetrics(predicted, actual, lab_thresh=-100, pred_thresh=-100) 
        self.assertEqual(metrics[0], 1.0)
        self.assertTrue(math.isnan(metrics[1]))
        self.assertEqual(metrics[2], 1.0)
        self.assertTrue(math.isnan(metrics[3]))
        self.assertEqual(metrics[4], 0)
        self.assertTrue(math.isnan(metrics[5]))

class MainFunctionsTests(unittest.TestCase):
    def testModelEvaluation(self):
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
        lossfn = pearsonCorrLoss 
        test_results = test(sample_size=50, model=model, loadName="models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt", validation_generator=validation_generator, lossfn=lossfn,device=device) #iterate over a few random test images 
        self.assertTrue(test_results[0][0] > test_results[1][0]) #ml pearson model > null 
        self.assertTrue(test_results[2][0] > test_results[3][0]) #mse model > null 
        self.assertTrue(0 <= test_results[0][0] <= 1)
        self.assertTrue(0 <= test_results[2][0])
        self.assertEqual(len(set(train_indices).intersection(test_indices)), 0) #make sure train and test indices do not overlap

class DatasetTests(unittest.TestCase):
    def testImageDataset(self):
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
        ##iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            ##make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ##make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 2, 2048, 2048))
            self.assertEqual(tuple(local_labels.shape), (1, 2048, 2048))
            i += 1
            if i > 10:
                break

    def testOsteosarcomaDataset(self):
        csvName = "/home/wongd/phenoScreen/datasets/cyclin_dataset.csv"
        dataset = OsteosarcomaDataset(csvName)
        generator = data.DataLoader(dataset, sampler = SubsetRandomSampler(list(range(0, len(dataset)))))
        i = 0
        ##iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            ##make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ##make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 1, 1104, 1104))
            self.assertEqual(tuple(local_labels.shape), (1, 1104, 1104))
            i += 1
            if i > 10:
                break

    def testOsteosarcomaAblationDataset(self):
        csvName = "/home/wongd/phenoScreen/datasets/cyclin_dataset.csv"
        dataset = OsteosarcomaAblationDataset(csvName, thresh_percent=1.0) #full ablation
        generator = data.DataLoader(dataset, sampler = SubsetRandomSampler(list(range(0, len(dataset)))))
        i = 0
        ##iterate over a random subset of our data to test 
        for names, local_batch, local_labels in generator:
            ##make sure data range is bounded correctly
            self.assertTrue(0 <= torch.max(local_batch) <= 255)
            ##make sure inputs and labels are correctly shaped
            self.assertEqual(tuple(local_batch.shape), (1, 1, 1104, 1104))
            self.assertEqual(tuple(local_labels.shape), (1, 1104, 1104))
            ##make sure all of input is ablated
            self.assertEqual(np.count_nonzero(local_batch.cpu().numpy()), 0)
            i += 1
            if i > 10:
                break




unittest.main()
