from utils import *
from sklearn.datasets import make_blobs



if __name__ == '__main__':
    
    
    dataset = fvecs_read(r"My-code\sift\sift_learn.fvecs")
    sub_dataset = dataset[:200, :2]
    #draw_vectors(sub_dataset)

