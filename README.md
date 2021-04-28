# Predicting Lane Change Decision Making with Compact Support
Dataset and code for IEEE IV21 paper Predicting Lane Change Decision Making with Compact Support

## Dataset
Data extraction process:
1. Use extract_lane_change.py to extract lane change sequences from both dataset, and extract_lane_change_merge_after.py to extract lane change abortions.
2. Use extract_sample.py and extract_sample_merge_after.py to extract the exact snapshot from the lane change sequences for the downstream prediction task.
3. Use data_preprocess.py to assemble the extracted samples; split to training and testing; generate OOD samples.

 Dataset | Description | Generating program | 
----|----|----
 us80.npz| Raw lane changes from US80, each row is one sample, with record [merge in front/after(0/1), u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y].| extract_sample.py, extract_sample_merge_after.py, and preprocess_both_dataset() in data_preprocess.py
 us101.npz| Raw lane changes from US101, each row is one sample, with record [merge in front/after(0/1), u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]| extract_sample.py, extract_sample_merge_after.py, and preprocess_both_dataset() in data_preprocess.py|
samples_relabeled_by_decrease_in_dx.npz| Samples with label based on $\Delta x$, each row is a sample with record [index, dt, u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]| extract_samples_relabel_as_change_in_distance() in extract_sample.py |
combined_dataset.npz| Combine us80 and us101 datasets, do feature selection, and normalize them to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test, f['e']=xGenerated), each sample is of feature x = [dv0, dv1, dx0, dx1] | prepare_validate_and_generate_ood() in data_process.py |
combined_dataset_before_feature_selection.npz| Combine us80 and us101 datasets and normalize them to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test), each sample is of feature x = [v_ego, dv0, dv1, dv2, dx0, dx1, dx2, dy0, dy1, dy2]| prepare_validate_and_feature_selection() in data_process.py|
combined_dataset_trainUs80_testUs101.npz| us80 as training dataset, us101 as testing dataset. Normalize samples to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test, f['e']=xGenerated), each sample is of feature x = [dv0, dv1, dx0, dx1] | prepare_validate_and_generate_ood_trainUs80_testUs101() in data_process.py|
combined_dataset_trainUs101_testUs80.npz| us101 as training dataset, us80 as testing dataset. Normalize samples to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test, f['e']=xGenerated), each sample is of feature x = [dv0, dv1, dx0, dx1] | prepare_validate_and_generate_ood_trainUs80_testUs101() in data_process.py|
