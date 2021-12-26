from .dataset import DatasetFromList, DatasetFromSessionList


def get_training_set(data_path, train_list, crop_size=[256, 256], left_right=False, kitti=False, kitti2015=False, airsim=False, shift=0, scale_factor=1.0):
  if airsim:
    return DatasetFromSessionList(train_list, crop_size, True, left_right, airsim, shift, scale_factor)
  else:
    return DatasetFromList(data_path, train_list,
                           crop_size, True, left_right, kitti, kitti2015, shift)


def get_test_set(data_path, test_list, crop_size=[256, 256], left_right=False, kitti=False, kitti2015=False, airsim=False, scale_factor=1.0):
  if airsim:
    return DatasetFromSessionList(test_list, crop_size, False, left_right, airsim, scale_factor)
  else:
    return DatasetFromList(data_path, test_list,
                           crop_size, False, left_right, kitti, kitti2015)
