'''
    author : Sachin kumar
    roll no: 108118083
    domain : Signal Processing and ML
    subdomain : Machine Learning
'''

import numpy as np 
import pandas as pd 
import os


class NeuralNetwork():
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, directory=None):
        '''
            input_layer_size : no. of units in the input layer
            hidden_layer_size : no.of units in hidden layer
            ouput_layer_size : no. of units in ouput layer
            directory : directory in which to save trained models weights 
        '''
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.directory = directory
    
    def train(self, X, Y, learning_rate, epochs, initial_iteration=0, lambda_=1, weights_directory=None):
        '''
            X : training data
            Y : training labels
            learning rate : Learning rate for Gradient Descent
            epochs : number of times the model sees the whole dataset
            initial_iteration : iteration to start from
            lambda : regularisation parameter
            file_name : file name where weigths are saved, if loading weights
        '''

        print('shape of training dataset',X.shape)
        print('input layer size =',self.input_layer_size)
        print('hidden layer size =',self.hidden_layer_size)
        print('output layer size =',self.output_layer_size)

        if initial_iteration==0:
            #Initializing random weights 
            Theta1 = self.randInitializeWeights(self.input_layer_size, self.hidden_layer_size)
            Theta2 = self.randInitializeWeights(self.hidden_layer_size, self.output_layer_size)
            loss_list = []
        else:
            #If continuing the training, then load the weights from weights_directory
            theta1_df = pd.read_csv(f'{weights_directory}/Theta1_{initial_iteration}.csv')
            theta2_df = pd.read_csv(f'{weights_directory}/Theta2_{initial_iteration}.csv')
            loss_df = pd.read_csv(f'{weights_directory}/Loss_list_{initial_iteration}.csv')
            print('Continued from iteration',initial_iteration)

            #converting dataframe to ndarray
            Theta1 = theta1_df.values[:,1:]  #avoiding that first irrelevant coloumn
            Theta2 = theta2_df.values[:,1:]
            loss_list = loss_df.values[:,1:].reshape(loss_df.shape[0]).tolist()  #ndarray to list


        loss_list, curr_irr, Theta1, Theta2 = self.Gradient_Descent(X, Y, Theta1, Theta2, loss_list, 
                                                               learning_rate, epochs, initial_iteration, lambda_)
        
        #setting the weights and loss_list
        self.Theta1 = Theta1
        self.Theta2 = Theta2
        self.loss_list = loss_list

        #returns the training loss
        n = len(loss_list)
        training_loss = loss_list[n-1]
        return training_loss
    
    def Gradient_Descent(self, X, Y, Theta1, Theta2, loss_list, learning_rate, epochs, initial_iteration, lambda_):
        '''
            Parameters:

            X : (ndarray) training data
            Y : (ndarray) training labels
            Theta1, Theta2 : Neural networks weights
            loss_list : List containing loss over epochs
            learning_rate : Learning rate of Gradient Descent
            epochs : number of times the model sees the whole dataset
            initial_iteration : iteration to start from
            lambda : regularisation parameter

            Returns:
            
            Theta1, Theta2 : Neural networks weights after training
            loss_list : List containing loss over epochs after training
        '''

        #helper variable to allow stop training
        interupt = initial_iteration + 5  
        curr_irr = 0

        #looping over epochs
        for i in range(initial_iteration,initial_iteration + epochs):

            #training the neural net 
            loss, Theta1_grad, Theta2_grad = self.backPropagation(X, Y, Theta1, Theta2, lambda_)
            loss_list.append(loss)

            #stop training if Loss increases
            if i>=1 and loss_list[i]>loss_list[i-1]:
                print('Loss increasing')
                break

            #Updating the weights
            Theta1 = Theta1 - learning_rate*Theta1_grad
            Theta2 = Theta2 - learning_rate*Theta2_grad

            print('epoch',i+1,'completed. loss =',loss)

            # Save the progress
            if (i+1)==interupt:
                interupt += 5

                #Saving the files
                self.save_file(loss_list, Theta1, Theta2, i+1)

            curr_irr = i+1

        return loss_list, curr_irr, Theta1, Theta2
    
    #We will create an algorithm independent of this particular application, a general backpropagation algorithm
    def backPropagation(self,X, Y, Theta1, Theta2, lambda_=1):
        m = X.shape[0]
        J = 0                           #J is our loss value

        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)

        #looping over training examples
        for i in range(m):
            
            #obtaining activations of each layer by forward propagation
            z_2, z_3, a_1, a_2, a_3 = self.forward_propagation(X[i], Theta1, Theta2)

            #calculating the error vectors Delta3 and Delta2
            delta_3 = a_3 - Y[i]
            delta_2 = np.dot(Theta2[:,1:].T, delta_3)*sigmoidGradient(z_2)

            #Calculating the Gradient of weight matrices
            Theta1_grad = Theta1_grad + np.dot(delta_2.reshape(delta_2.shape[0],1), 
                                               a_1.reshape(a_1.shape[0],1).T)
            Theta2_grad = Theta2_grad + np.dot(delta_3.reshape(delta_3.shape[0],1),
                                               a_2.reshape(a_2.shape[0],1).T)

            #calculating loss
            J += np.sum((Y[i]-a_3)**2)


        #calculating regularised cost function 
        J = J/m + (lambda_/(2*m))*(np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))

        D_1 = np.hstack( ((1/m)*Theta1_grad[:,0:1], 
                       (1/m)*(Theta1_grad[:,1:] + lambda_*Theta1[:,1:])))

        D_2 = np.hstack(((1/m)*Theta2_grad[:,0:1], 
                       (1/m)*(Theta2_grad[:,1:] + lambda_*Theta2[:,1:])))

        return J,D_1,D_2

    def forward_propagation(self,x, Theta1, Theta2):
        '''
            Parameters:
            x : vector of size input_layer_size, to be fed into neural network
            Theta1, Theta2 : weights of the neural network

            Returns:
            z_2, z_3 : Intermediate vectors calculated during forward propagation 
            a_1, a_2 : activation of input, hidden and ouput layer
            a_3 : ouput of neural network
        '''
        a_1 = np.hstack((np.ones((1)),x))         # added bias to input layer
        z_2 = np.dot(Theta1,a_1)
        a_2 = np.hstack((np.ones(1), sigmoid(z_2)))  # added bias to hidden layer, and sigmoid activation applied
        z_3 = np.dot(Theta2, a_2)
        a_3 = z_3   #not applying sigmoid here

        return z_2, z_3, a_1, a_2, a_3
        
    
    def predict(self, X, Y):
        '''
            Parameters:
            X, Y - Test data, and Test label

            Returns:
            prediction - Predicted output
            mse - mean squared error of predictions
        '''
        n = X.shape[0]
        prediction = []
        mse = 0
        #looping over test examples
        for i in range(n):
            #calculating ouput for current test example by forward propagation
            _, _, _, _, pred = self.forward_propagation(X[i], self.Theta1, self.Theta2)
            #squared error of current prediction
            mse += np.sum((pred-Y[i])**2) 
            #adding current prediction to the list
            prediction.append(pred)
        
        #mean of squared errors
        mse = mse/n
        return prediction, mse


    def randInitializeWeights(self, L_in, L_out):
        '''
            Parameters:
            L_in : Numbers of units in the layer to which the returned 
                    matrix acts upon
            L_out : number of units in the layer whose activations are
                    calculated using the returned weight matrix 

            Returns :
            Theta : matrix with with shape (L_out,L_in+1) and random values
        '''
        #initialising epsilon
        epsilon_init = np.sqrt(6)/(L_in + L_out)
        #initializing random matix with shape (L_out,L_in+1)
        #each value in marix belongs to range (-epsilon, epsilon)
        Theta = np.random.rand(L_out, L_in +1)*2*epsilon_init - epsilon_init
        return Theta

    def save_file(self,loss_list, Theta1, Theta2, itr):
        '''
            helper function to save the intermediate weights while calculation

        '''
        #save files in drive for later use
        df_Theta1 = pd.DataFrame(Theta1)
        df_Theta2 = pd.DataFrame(Theta2)
        df_loss_list = pd.DataFrame(loss_list)

        file_name = f'iteration {itr}'
        directory = self.directory

        os.chdir(directory)
        try:
            os.mkdir('{}'.format(file_name))
        except:
            pass

        df_Theta1.to_csv('{}/{}/Theta1_{}.csv'.format(directory, file_name, itr))
        df_Theta2.to_csv('{}/{}/Theta2_{}.csv'.format(directory, file_name, itr))
        df_loss_list.to_csv('{}/{}/Loss_list_{}.csv'.format(directory, file_name, itr))

        os.chdir('/content')

        print('All files saved successfully')   


#Helper functions
def sigmoid(z):
        #returns the sigmoid of a matrix,vector or scalar
        return 1/(1 + np.exp(-z))

#calculates the sigmoid Gradient of scalar, vector, matrix or tensor
def sigmoidGradient(z):
    return sigmoid(z)*(1 - sigmoid(z))
