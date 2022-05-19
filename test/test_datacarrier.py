import shutil
import unittest
import os
import numpy as np
import torch

from light_seg.data_carrier import DataCarrier
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor


class DataCarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.random.rand(16, 32, 32)
        self.label = (self.image > 0.5).astype(int)
        self.data_dir = "tmp_test"
        self.image_dir = os.path.join(self.data_dir, "imagesTr")
        self.label_dir = os.path.join(self.data_dir, "labelsTr")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        np.save(os.path.join(self.image_dir, "test.npy"), self.image)
        np.save(os.path.join(self.label_dir, "test.npy"), self.label)
        return

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir)
        return

    def test_get_data_samples(self):
        datacarrier = DataCarrier()
        samples = datacarrier.get_data_samples(
            self.data_dir, slice_offset=0, slice_axis=0
        )
        self.assertEqual(len(samples), self.image.shape[0])
        self.assertEqual(
            samples[0]["image_path"], os.path.join(self.image_dir, "test.npy")
        )
        self.assertEqual(
            samples[0]["label_path"], os.path.join(self.label_dir, "test.npy")
        )
        self.assertEqual(samples[0]["slice_idx"], 0)

    def test_load_slice(self):
        datacarrier = DataCarrier()
        samples = datacarrier.get_data_samples(
            self.data_dir, slice_offset=0, slice_axis=0
        )
        slice = datacarrier.load_slice(samples[0])
        np.testing.assert_array_equal(
            slice["data"][0], np.expand_dims(self.image[0], axis=0)
        )
        np.testing.assert_array_equal(
            slice["seg"][0], np.expand_dims(self.label[0], axis=0)
        )
        self.assertEqual(
            slice["image_paths"][0], os.path.join(self.image_dir, "test.npy")
        )
        self.assertEqual(
            slice["label_paths"][0], os.path.join(self.label_dir, "test.npy")
        )
        self.assertEqual(slice["slice_idxs"][0], 0)

    def test_concat_data(self):
        datacarrier = DataCarrier()
        samples = datacarrier.get_data_samples(
            self.data_dir, slice_offset=0, slice_axis=0
        )
        transforms = Compose(
            [
                NumpyToTensor(),
            ]
        )
        for sample in samples:
            slice = datacarrier.load_slice(sample)
            slice = transforms(**slice)
            image_tensor = slice["data"].squeeze()
            ones = torch.full(image_tensor.size(), 0.9)
            zeros = torch.full(image_tensor.size(), 0.1)
            label_c0 = torch.where(image_tensor > 0.5, zeros, ones).unsqueeze(0)
            label_c1 = torch.where(image_tensor > 0.5, ones, zeros).unsqueeze(0)
            softmax_pred = torch.cat((label_c0, label_c1), dim=0).unsqueeze(0)
            datacarrier.concat_data(slice, softmax_pred)
        np.testing.assert_array_equal(
            datacarrier.data[os.path.join(self.image_dir, "test.npy")]["data"],
            self.image,
        )
        np.testing.assert_array_equal(
            datacarrier.data[os.path.join(self.image_dir, "test.npy")]["seg"],
            self.label,
        )
        np.testing.assert_array_equal(
            datacarrier.data[os.path.join(self.image_dir, "test.npy")]["pred_seg"],
            self.label,
        )


if __name__ == "__main__":
    unittest.main()
