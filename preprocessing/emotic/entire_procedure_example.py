from tqdm import tqdm
from preprocessing.annotations_browser import AnnotationsBrowser


matfile='/home/sandbox/Desktop/EMOTIC_database/annotations/annotations/Annotations.mat'
EMOTIC_base_dir='/home/sandbox/Desktop/EMOTIC_database/emotic'
mode='test'




# First create an instance of the AnnotationsBrowser class
browser = AnnotationsBrowser(matfile=matfile,
                             EMOTIC_base_dir=EMOTIC_base_dir,
                             mode=mode)

if browser.mode == 'train':
    nb_samples = 17077
elif browser.mode == 'val':
    nb_samples = 2088
elif browser.mode == 'test':
    nb_samples = 4389



copy_entire_single_occurrence_imgs_dir = '/home/sandbox/Desktop/EMOTIC_database/entire_single_occurrence_imgs/'+ browser.mode + '/'
copy_entire_multiple_imgs_dir = '/home/sandbox/Desktop/EMOTIC_database/entire_multiple_imgs/'+ browser.mode + '/'


copy_dir = dir_name + 'images/'
# emotion_categories_filename = dir_name + 'emotions.csv'

for field_number in tqdm(range(0, nb_samples)):
    browser.copy_images(field_number=field_number, copy_dir=dir_name)
