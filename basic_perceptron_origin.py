import numpy as np

number_of_features = 2 #... 
number_of_times_to_cycle_thru_data = 10 #... 
all_xs = [[-1,-1],[1,0],[-1,10]] #... # assume is length-N list of vectors, each of dimension `number_of_features` 
all_ys = [1,-1,1] #... # assume is length-N list of signs (+1s, -1s) and that the nth entry matches all_xs's nth entry 
testing_x = [1,1] #... # a testing vector of dimension `number_of_features`

predict = lambda w,x : np.sign(w.dot(x)) 

def find_w_using_perceptron(sequence_of_training_points_to_consider): 
    w = 0. * np.zeros(number_of_features) 
    for x,y in sequence_of_training_points_to_consider: 
        #x,y = fetch_training_point() 
        if y==predict(w,x): 
            continue 
        w += y*x 
    return w

sequence_of_training_points_to_consider = ((x,y) for _ in range(number_of_times_to_cycle_thru_data) for x,y in zip(all_xs, all_ys))

learned_hypothesis = find_w_using_perceptron(sequence_of_training_points_to_consider)

prediction_for_testing_vector = predict(learned_hypothesis, testing_x)