import unittest
from src.ml.dataset import load_data

class TestDataset(unittest.TestCase):

    def test_load_data(self):
        # Test loading the dataset from the CSV and image directories
        data, labels = load_data('data/fall_dataset/data.csv', 'data/fall_dataset/images')
        self.assertIsNotNone(data)
        self.assertIsNotNone(labels)
        self.assertEqual(len(data), len(labels))

    def test_data_shape(self):
        # Test the shape of the loaded data
        data, labels = load_data('data/fall_dataset/data.csv', 'data/fall_dataset/images')
        self.assertEqual(data.shape[1], expected_feature_count)  # Replace with actual expected feature count
        self.assertEqual(len(labels), len(data))

if __name__ == '__main__':
    unittest.main()