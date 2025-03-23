import numpy as np 
import json
import copy 



###################################################Neural Network#################################################################
def print_graphe_graphviz(SSize,w,b,X,results) : 
    to_print = "digraph G {\nnode [shape=record,width=.1,height=.1];\n"
    k = 0
    to_print = to_print + "subgraph cluster0 {\n"
    for i in range(len(X)):
        to_print = to_print + """node%d [label = "{<n>%d|%.2f|a<p>}"];\n""" % (k,i,X[i])
        k += 1
    to_print = to_print + "}\n"

    for i in range(len(SSize)-1) :
        to_print = to_print + "subgraph cluster%d {\n" % (i+1)
        for j in range(SSize[i+1]):
            to_print = to_print + """node%d [label = "{<n>%d/%d|%.2f|%.2f|a<p>}"];\n""" % (k,i,j,b[i][j],results[i][j])
            k += 1
        to_print = to_print + "}\n"
    
    totSSize = 0
    for i in range(len(SSize)-1) :
        for j in range(SSize[i]):
            for l in range(SSize[i+1]):
                to_print = to_print + """node%d -> node%d [label = %.2f];\n""" % (j+totSSize,l+totSSize+SSize[i],w[i][j][l])
        totSSize += SSize[i]
    to_print = to_print+ "}"
    return to_print

class neural_network(object):
    """
    ### crée un réseau de neurone : 

    `SSize` = [#input_layer, #hidden_layer_1,..., #hidden_layer_n, #output_layer]

    /!\ peut ne pas avoir d'hidden layer

    `LR`(Learning rate) ; LR = 1 par default 

    `act` (activation function) ; act = "sigmoid" par defaut ; (pour l'instant seulement sigmoid and ReLU)
    """
    def __init__(self,SSize,LR = 1,act = "sigmoid"):
        self.Size = SSize
        self.ActivationType = act

        self.w = [] #les poids/ taille : (nombre_de_couches-1)*(taille de la couche i)*(taille de la couche i+1)
        self.bias = [] #les correctifs d'érreurs / taille : (nb_couches-1)*(taille de la couche i)
        for i in range(len(self.Size)-1) : 
            self.w.append(np.random.randn(self.Size[i],self.Size[i+1])) #on met les poids en random 
            self.bias.append(np.random.randn(self.Size[i+1])) #on assigne les correctifs de chaques neurones 
        #self.bias.append(np.random.randn(self.Size[-1])) #on assigne les correctifs de la derniere couche 
        
        
        self.learningRate = LR 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []
        self.dError  = []


    #-------------------------------fonctions d'activations--------------------------------------

    def identite(self,s):
        return np.array(s)

    def identiteDerive(self,s):
        return np.array([1 for i in s])

    def ReLU(self,s):
        return np.array([i if i >= 0 else 0.1*i for i in s ])

    def ReLUDerive(self,s):
        return np.array([1 if i >= 0 else 0.1 for i in s ])

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidDerive(self,s):
        return self.sigmoid(s)*(1-self.sigmoid(s))

    def tanh(self,s):
        return (2/(1+np.exp(-2*s)))-1

    def tanhDerive(self,s):
        return 1-(self.tanh(s)*self.tanh(s))

    def activation(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoid(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLU(s)
        elif self.ActivationType == "id" :
            return self.identite(s)
        elif self.ActivationType == "tanh" :
            return self.tanh(s)

    def activationDerive(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoidDerive(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLUDerive(s)
        elif self.ActivationType == "id" :
            return self.identiteDerive(s)
        elif self.ActivationType == "tanh" :
            return self.tanhDerive(s)
        
    
    #-------------------evaluation de la fonction reseau de neurones en X--------------------------------
    def forward(self,X):
        #self.preAct = 0 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []

        self.preAct.append(np.dot(X,self.w[0]) + self.bias[0]) #produit matricielle 
        self.results.append(self.activation(self.preAct[0])) #activation du neurone (lissage des données)
        
        for i in range(1,len(self.Size)-2) : #on itere pour chaque neurone 
            self.preAct.append(np.dot(self.results[i-1],self.w[i]) + self.bias[i])
            self.results.append(self.activation(self.preAct[i]))
        
        self.preAct.append(np.dot(self.results[len(self.Size)-3],self.w[len(self.Size)-2]) + self.bias[len(self.Size)-2])
        self.results.append(self.preAct[len(self.Size)-2])

        return self.results[-1]


    #---------------------------Entrainement--------------------------------------

    def backward(self,X,y,o):
        """
        X : valeur d'entree 

        y : valeur attendu par le modele 

        o : valeur retourne par le model pour une entree donne
        """
        self.dError = [] #calcul de l'erreur 


        list.insert(self.dError,0, 2*(o-y))
        for i in range(1,len(self.Size)-1) : 
            list.insert(self.dError,0, np.dot(self.dError[0],self.w[-i].T) * self.activationDerive(self.preAct[-(i+1)]))

        self.w[0] -= np.dot(np.array(X,ndmin=2).T, np.array(self.dError[0],ndmin=2))*self.learningRate
        self.bias[0] -= self.dError[0]*self.learningRate
        

        for i in range(1,len(self.Size)-1) : 
            self.w[i] -= np.dot(np.array(self.results[i-1],ndmin=2).T, np.array(self.dError[i],ndmin=2))*self.learningRate
            self.bias[i] -= self.dError[i]*self.learningRate
        

    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)
        return np.abs(o-y)


    def error(self,X,y):
        o = self.forward(X)
        return np.abs(o-y)


    def predict(self,xIncunue):
        print(xIncunue)
        print(tuple(self.forward(xIncunue)))



    #--------------------------sauvegarder, charger et afficher le reseau de Neurones-----------------------------
    def print_NN(self,X):
        o = self.forward(X)
        print(print_graphe_graphviz(self.Size,self.w,self.bias,X,self.results))

    def save_NN(self, name = "Neural_Network_save"):
        with open(name+'.json', 'w', encoding='utf-8') as f:
            
            w2 = [self.w[i].tolist() for i in range(len(self.w))]
            b2 = [self.bias[i].tolist() for i in range(len(self.bias))]
            activation_f = self.ActivationType
            dic = {"weight" : w2, "bias" : b2,"act_f" : activation_f}
            json.dump(dic, f, ensure_ascii=False, indent=4)  

    def load_NN(self,name):
        with open(name+'.json') as f:
            data_loaded = json.load(f)
            self.w = [np.array(data_loaded["weight"][i]) for i in range(len(data_loaded["weight"]))]
            self.bias = [np.array(data_loaded["bias"][i]) for i in range(len(data_loaded["bias"]))]
            self.ActivationType = data_loaded["act_f"]


def print_shape(array):
    for i in range(len(array)):
        print(array[i].shape)
####################################################Deep-Qlearning############################################################

import pygame
import sys
import random
import time

import matplotlib.pyplot as plt

"""
regles du jeu, on place un cavalier sur un plateau, on place une pomme et le but c'est d'y arriver avec le moins de coup possible

"""


class data_analyser : 
    """
    prend des données, les synthetises et on peut faire des truc avec 
    """
    def __init__(self) -> None:
        self.data = []
        self.taille = 0

    def add(self,entier):
        self.data.append(entier)
        self.taille += 1

    def moyenne_partielle(self,n): #attention n<= taille
        moy_part = []
        if n <= self.taille :
            for i in range(self.taille-n):
                moy_part.append(0)
                for j in range(n):
                    moy_part[i] += self.data[i+j]
                moy_part[i] = moy_part[i]/n
        
        return moy_part
    
    def show_moy_part(self,n): #attention n<= taille
        X = [i for i in range(self.taille-n)]
        Y = self.moyenne_partielle(n)
        plt.plot(X,Y)
        plt.show()        

    """ def motenne_par_parties(self,nb_parties): #nb_parties <= taille
        moy_pp = []
        taille_parties = self.taille//nb_parties
        rest = self.taille%nb_parties
        
        for i in range(nb_parties-1):
            moy_pp.append(0)
            for j in range(taille_parties):
                moy_pp[i] =  """


class machin:
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

class plateau : 
    def __init__(self,i_depart,j_depart, width = 8, height = 8) -> None:

        self.width = width
        self.height = height

        self.cavalier = machin(i_depart,j_depart)

        self.pomme = machin(np.random.randint(0,8), np.random.randint(0,8))
        while self.pomme.x == self.cavalier.x and self.pomme.y == self.cavalier.y :
            self.pomme.x = np.random.randint(0,8)
            self.pomme.y = np.random.randint(0,8)
        
        self.is_win = False

    def test_win(self) -> bool:
        return self.pomme.x == self.cavalier.x and self.pomme.y == self.cavalier.y
    

    def coup_possibles(self) -> list :
        actions = []
        tout_les_coups = [(0,2,1),(1,1,2),(2,-1,2),(3,-2,1),(4,-2,-1),(5,-1,-2),(6,1,-2),(7,2,-1)]
        for index,coupx,coupy in tout_les_coups:
            if self.cavalier.x+coupx < 8 and self.cavalier.x+coupx >= 0 and self.cavalier.y+coupy < 8 and  self.cavalier.y+coupy >= 0 : 
                actions.append((index,coupx,coupy))
        return actions

    def update(self,coup): #on suppose qu'on ne joue que des coup legaux 
        if self.is_win == False : 
            self.cavalier.x += coup[1]
            self.cavalier.y += coup[2]

            self.is_win = self.test_win()

            if self.is_win :
                print("the horse ate the apple")

        else :
            print("there is no apple left")

    def init_draw(self,size):
        pygame.init()
        self.surface = pygame.display.set_mode((size+40,size+40))
        self.screen_size = size

        self.imgpomme = pygame.image.load("pngegg.png")
        self.imgpomme = pygame.transform.scale(self.imgpomme, (self.screen_size/8, self.screen_size/8))

        self.imgcavalier = pygame.image.load("cacacavalier.png")
        self.imgcavalier = pygame.transform.scale(self.imgcavalier, (self.screen_size/8, self.screen_size/8))


    
    def draw(self):
        self.surface.fill((255,255,255))

        for i in range(8):
            for j in range(8):
                if (i+j)%2 == 0 :
                    pygame.draw.rect(self.surface,(255,255,255),pygame.Rect(j*self.screen_size/8+20,i*self.screen_size/8+20,self.screen_size/8,self.screen_size/8))
                else : 
                    pygame.draw.rect(self.surface,(0,0,0),pygame.Rect(j*self.screen_size/8+20,i*self.screen_size/8+20,self.screen_size/8,self.screen_size/8))

        pygame.draw.rect(self.surface,(0,0,0),pygame.Rect(19,19,self.screen_size+2,self.screen_size+2),1)

        font = pygame.font.SysFont(None, 24)
        for i in range(8):
            img = font.render(chr(ord('h')-i), True, (0,0,0))
            self.surface.blit(img, (5, 20+(i+1/2)*self.screen_size/8))
            self.surface.blit(img, (25+self.screen_size, 20+(i+1/2)*self.screen_size/8))

        for j in range(8):
            img = font.render(str(j+1), True, (0,0,0))
            self.surface.blit(img, (20+(j+1/2)*self.screen_size/8,5))
            self.surface.blit(img, (20+(j+1/2)*self.screen_size/8,25+self.screen_size))

        self.surface.blit(self.imgpomme,(self.pomme.x*self.screen_size/8+20,self.pomme.y*self.screen_size/8+20))
        self.surface.blit(self.imgcavalier,(self.cavalier.x*self.screen_size/8+20,self.cavalier.y*self.screen_size/8+20))

        

                          
        pygame.display.flip()
        

    def read(self):
        return [self.cavalier.x, self.cavalier.y, self.pomme.x-self.cavalier.x, self.pomme.y-self.cavalier.y]


    def argmax_coup(self,actions):
        val_maxi = float("-inf")
        for index,coupx,coupy in self.coup_possibles() :
            if actions[index] > val_maxi : 
                val_maxi = actions[index]
                max_coup = (index,coupx,coupy)
        return max_coup
    
    def valmax_coup(self,actions):
        val_maxi = float("-inf")
        for index,coupx,coupy in self.coup_possibles() :
            if actions[index] > val_maxi :
                val_maxi =  actions[index]
        return val_maxi


    def get_reward(self):
        if self.is_win : 
            return 100
        else :
            return -1
        
    def copy(self):
        b2 = plateau(self.cavalier.x, self.cavalier.y, self.width, self.height)
        b2.pomme.x = self.pomme.x
        b2.pomme.y = self.pomme.y
        b2.is_win = self.is_win
        return b2

def DQL():

    data_learning = data_analyser()

    gamma = 0.9

    Q  = neural_network([4,50,50,50,8],LR = 0.0001,act = "sigmoid")
    # Qt = neural_network([4,16,16,8],LR = 0.1,act = "sigmoid")
    # Qt.w = copy.deepcopy(Q.w)
    # Qt.b = copy.deepcopy(Qt.b)

    N = 100000
    Memory = 200
    D = []
    board = plateau(np.random.randint(0,8), np.random.randint(0,8))
    board.init_draw(800)
    for i in range(N):
        print("tour n°",i+1)
        jj = 0 
        while (not board.is_win) and jj <= 10 :
            time.sleep(0.1)
            
            jj = jj+1

            eps = (i+1)**(-1)

            pre = board.read()
            if np.random.random() <= eps:
                coups = board.coup_possibles()
                i = np.random.randint(0,len(coups))
                c = coups[i]
            else :
                actions = Q.forward(pre)
                #print(actions)
                c = board.argmax_coup(actions)

            #print("before :",board.cavalier.x,board.cavalier.y,c)
            board.update(c)
            #print("after :",board.cavalier.x,board.cavalier.y)
            
            r = board.get_reward()
            post = board.copy()
            is_won = board.is_win

            D.append((pre,c,r,post))

            train_sample = random.sample(D,min(10,len(D)))

            for pre,c,r,post in train_sample:
                if post.is_win :
                    y = r
                else :
                    y = gamma*post.valmax_coup(Q.forward(post.read()))+r
                
                o = Q.forward(pre)
                vy = o.copy()
                vy[c[0]] = y
                Q.backward(pre,vy,o)

            board.draw()


        if len(D) >= Memory :
            D.pop(0)
            
        print("fin en",jj,"coups\n--------------------")
        data_learning.add(jj)
        pygame.quit()
        board = plateau(np.random.randint(0,8), np.random.randint(0,8))
        board.init_draw(800)
        board.draw()

    print("vraiment fini")

    pygame.quit()

    Q.save_NN("Trojean_ttt")

    data_learning.show_moy_part(100)
    sys.exit() 




def test_model():

    Q = neural_network([4,50,50,50,8],LR = 0.0001,act = "sigmoid")
    Q.load_NN("Trojean_long")
    data_learning = data_analyser()

    board = plateau(np.random.randint(0,8), np.random.randint(0,8))
    #board.init_draw(800)

    ii = 0 
    while ii <= 10000 : 
        ii += 1
        if ii%100 == 0 : 
            print(ii)
        
        jj = 0 
        while (not board.is_win) and jj <= 25 :
            
            jj = jj+1

            actions = Q.forward(board.read())
            c = board.argmax_coup(actions)
            board.update(c)

            
            
            #board.draw()
            #time.sleep(10)
        data_learning.add(jj)

        #pygame.quit()
        board = plateau(np.random.randint(0,8), np.random.randint(0,8))
        #board.init_draw(800)
        #board.draw()
        
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        pygame.quit()
        #        sys.exit()
    plt.ylim(5, 25)
    data_learning.show_moy_part(1000)


            




            






if __name__ == "__main__" : 
    """run = True
    board = plateau(0,0)

    board.init_draw(800)
    moves = board.coup_possibles()

    while run:
        board.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit() """
    DQL()
    ##test_model()