import argparse
from argparse import ArgumentParser
import SimpleITK as sitk
import numpy
from stl import mesh
import numpy as np
from stltovoxel.convert import convert_meshes
from scipy.ndimage import gaussian_filter


def main_cli():
    parser = ArgumentParser()

    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+',
        help="The input stl files that should be parsed to nifti files"
    )

    parser.add_argument(
        "--save_file_name",
        type=str,
        help="The name of the resulting nifti file"
    )

    parser.add_argument(
        "--object_size",
        type=int,
        help="The size of the object in pixels.",
        default=100
    )

    parser.add_argument(
        "--image_size",
        type=int,
        nargs='+',
        help="Size of the resulting nifti image. Can be one int (then x=y=z) or three ints for the three dimensions",
        default=[256]
    )

    parser.add_argument(
        "--object_offset",
        type=int,
        nargs='+',
        help="The offset where the object should be placed in the image. "
             "Can be one int (then the offset is the same in all 3 dimensions) or three ints for the three dimensions.",
        default=[50]
    )

    parser.add_argument(
        "--blur",
        help="Apply gaussian blur",
        action="store_true"
    )

    parser.add_argument(
        "--noise",
        help="Add noise in background",
        action="store_true"
    )

    args = parser.parse_args()

    if len(args.image_size) == 1:
        args.image_size = (args.image_size[0], args.image_size[0], args.image_size[0])
    elif len(args.image_size) == 3:
        args.image_size = tuple(args.image_size)
    else:
        raise argparse.ArgumentError(args.image_size, "Image size has to be exaclty 1 or 3 values")

    if len(args.object_offset) == 1:
        args.object_offset = (args.object_offset[0], args.object_offset[0], args.object_offset[0])
    elif len(args.object_offset) == 3:
        args.object_offset = tuple(args.object_offset)
    else:
        raise argparse.ArgumentError(args.object_offset, "Object offset has to be exaclty 1 or 3 values")

    return args


def meshes_to_numpy(file_names, resolution):
    meshes = []
    for file_name in file_names:
        mesh_obj = mesh.Mesh.from_file(file_name)
        org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
        meshes.append(org_mesh)
    vol, _, _ = convert_meshes(meshes, resolution, False)
    vol = vol.astype(numpy.float32)
    return vol


def embed_object_in_image(start_indices, object_arr, image_size):
    image = np.zeros(image_size)
    try:
        image[start_indices[0]:start_indices[0] + object_arr.shape[0],
        start_indices[1]:start_indices[1] + object_arr.shape[1],
        start_indices[2]:start_indices[2] + object_arr.shape[2]] = object_arr
        return image
    except:
        print("Index was out of bounds, could not fit object in image with given start indices.")
        return object_arr


def embed_object_in_image_offset_too_large(start_indices, object_arr, image_size):
    image = np.zeros(image_size)
    object_outside_0 = start_indices[0] + object_arr.shape[0] - image_size[0]
    object_outside_1 = start_indices[1] + object_arr.shape[1] - image_size[1]
    object_outside_2 = start_indices[2] + object_arr.shape[2] - image_size[2]
    end_index_0 = start_indices[0] + object_arr.shape[0] if object_outside_0 > 0 else \
        image_size[0]
    end_index_1 = start_indices[1] + object_arr.shape[1] if object_outside_1 > 0 else \
        image_size[1]
    end_index_2 = start_indices[2] + object_arr.shape[2] if object_outside_2 > 0 else \
        image_size[2]
    try:
        image[start_indices[0]:start_indices[0] + end_index_0,
        start_indices[1]:start_indices[1] + end_index_1,
        start_indices[2]:start_indices[2] + end_index_2] = object_arr[:object_arr.shape[0] - object_outside_0,
                                                                      :object_arr.shape[1] - object_outside_1,
                                                                      :object_arr.shape[2] - object_outside_2]
        return image
    except:
        print("Index was out of bounds, could not fit object in image with given start indices.")
        return object_arr

def embed_object_in_image_negative_offset(start_indices, object_arr, image_size):
    image = np.zeros(image_size)
    start_index_0 = start_indices[0] if start_indices[0] > 0 else 0
    start_index_1 = start_indices[1] if start_indices[1] > 0 else 0
    start_index_2 = start_indices[2] if start_indices[2] > 0 else 0
    obj_start_0 = 0 if start_indices[0] > 0 else abs(start_indices[0])
    obj_start_1 = 0 if start_indices[1] > 0 else abs(start_indices[1])
    obj_start_2 = 0 if start_indices[2] > 0 else abs(start_indices[2])
    try:
        image[start_index_0:start_indices[0] + object_arr.shape[0],
        start_index_1:start_indices[1] + object_arr.shape[1],
        start_index_2:start_indices[2] + object_arr.shape[2]] = object_arr[obj_start_0:, obj_start_1:, obj_start_2:]
        return image
    except:
        print("Index was out of bounds, could not fit object in image with given start indices.")
        return object_arr


def add_noise(noise_prob, image):
    prob_array = np.random.rand(image.shape[0], image.shape[1], image.shape[2])
    noise_array = np.random.rand(image.shape[0], image.shape[1], image.shape[2])
    noise_array[prob_array <= noise_prob] = 0
    image[image < 0.1] = noise_array[image < 0.1]
    return image


def numpy_to_nifti(np_array, save_file_name):
    img = sitk.GetImageFromArray(np_array)
    sitk.WriteImage(img, save_file_name)


if __name__ == "__main__":
    np.random.seed(22)
    args = main_cli()
    vol = meshes_to_numpy(args.input_files, args.object_size)
    vol = embed_object_in_image(args.object_offset, vol, args.image_size)
    if args.blur:
        vol = gaussian_filter(vol, sigma=8)
    if args.noise:
        vol = add_noise(0.5, vol)
    numpy_to_nifti(vol, args.save_file_name)
