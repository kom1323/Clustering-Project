from utils import *

parameters = {
    'eps': 262 / 2352,
    'k': 65,
    'b': 0.8,
    'num_iterations': 30
}





if __name__ == '__main__':
    
    
    dataset = fvecs_read(r"My-code\sift\sift_learn.fvecs")



    sub_dataset = dataset[:200, :2]

    draw_vectors(sub_dataset)