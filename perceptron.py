from numpy import random
import pandas as pd 
class Perceptron:
    def __init__(self, number_of_weights):
        self.number_of_weights = number_of_weights
        self.weights = self.generate_random_weights(number_of_weights)
        data = {'x': [1,1,0,0],
                     'y': [1,0,1,0],
                     'r': [1,0,0,0]}
        self.df = pd.DataFrame (data, columns = ['x','y','r'])
        

    def generate_random_weights(self,n):
        weights = []
        for x in range(n):
            weights.append(random.random()*10-5)
        return weights


    def prediccion(self,entrada1, entrada2):
        bias = self.weights[0]
        activation = (entrada1 * self.weights[1]) + (entrada2 * self.weights[2]) + bias
        return 1 if activation > 0 else 0

    def ajustarpesos(self,valorReal, rPred, e1, e2):
        self.weights[0] = self.weights[0] + valorReal
        self.weights[1] = self.weights[1] + rPred * e1
        self.weights[2] = self.weights[2] + rPred * e2

    def entrenamiento(self,epochs):
        for epoch in range(epochs):
            for i in range(0,4):
                valorReal = self.df.iloc[i]['r']
                e1 = self.df.iloc[i]['x']
                e2 = self.df.iloc[i]['y']
                resultadoPrediccion = self.prediccion(e1,e2)
                print("Resultado: " + str(resultadoPrediccion))
                if resultadoPrediccion != valorReal:
                    self.ajustarpesos(valorReal,resultadoPrediccion,e1,e2)
                    print('Adjusted weights: {}'.format(self.weights))
            print('Final weights from epoch {}: {}'.format(epoch,self.weights))

    def verify(self):
        count = 0
        for i in range(0,4):
            valorReal = self.df.iloc[i]['r']
            e1 = self.df.iloc[i]['x']
            e2 = self.df.iloc[i]['y']
            resultadoPrediccion = self.prediccion(e1,e2)
            if resultadoPrediccion != valorReal:
                count = count + 1
        return (1-count/len(self.df))*100

perceptron = Perceptron(3)
perceptron.entrenamiento(epochs=4)

print('\nPesos finales: {}'.format(perceptron.weights))
accuracy = perceptron.verify()
print('Error: {} %'.format(100-accuracy))
