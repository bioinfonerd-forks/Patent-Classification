import tarfile
import zipfile



path='C:\\Users\\fereshteh\\Documents\\Farshad - Thesis\\'

opener, mode = zipfile.ZipFile, 'r:gz'
tar=tarfile.open(path+'TFIDF_for_TF_matrix_after_removing_cutoff.tgz',mode)
tar.extractall()
tar.close()

