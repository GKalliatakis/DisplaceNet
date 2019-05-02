from hdf5_controller import Controller



hdf5_file_name = 'EMOTIC-VAD-Classification-rescale.hdf5'

train_csv_file_path ='/home/gkallia/git/emotic-VAD-classification/dataset/EMOTIC_resources/CSV/train.csv'
val_csv_file_path ='/home/gkallia/git/emotic-VAD-classification/dataset/EMOTIC_resources/CSV/val.csv'
test_csv_file_path ='/home/gkallia/git/emotic-VAD-classification/dataset/EMOTIC_resources/CSV/test.csv'

cropped_imgs_dir ='/home/gkallia/git/emotic-VAD-classification/dataset/EMOTIC_resources/cropped_imgs/'
entire_imgs_dir = '/home/gkallia/git/emotic-VAD-classification/dataset/EMOTIC_resources/entire_multiple_imgs/'
main_numpy_dir ='/home/gkallia/git/emotic-VAD-classification/dataset/EMOTIC_resources/numpy_matrices/'

controller = Controller(hdf5_file=hdf5_file_name,
                        train_csv_file_path=train_csv_file_path,
                        val_csv_file_path=val_csv_file_path,
                        test_csv_file_path=test_csv_file_path,
                        cropped_imgs_dir=cropped_imgs_dir,
                        entire_imgs_dir=entire_imgs_dir,
                        main_numpy_dir=main_numpy_dir)


create_hdf5 = controller.create_hdf5_VAD_regression(dataset='EMOTIC', input_size=224)
