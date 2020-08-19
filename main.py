import random
import math
import itertools
from scipy.stats import entropy as scipy_entropy
import numpy         as np 
import pandas        as pd
import random, copy
from ruamel_yaml import YAML

iteracion_max = 20
tam_poblacion = 80
num_torneo = 3
var_cruzamiento = 0.8
var_mutacion = 0.2
gen_cromosoma = 1

archivo               = "data.txt"
archivo               = open(archivo)
data                  = YAML().load(archivo.read())
archivo.close()


class cromosoma():
    def __init__(self, **kargs):
        data          = kargs.get("data", None)
        assert data  != None, "'data'"

        self.clase  = data["clase"]
        self.dia     = data["dia"] 
        self.horas    = data["horas"] 
        self.cols     = data["map_curso"] 
        self.room    = data["room"]
        
        # -- constants to reduce len() times --
        self.len_room   = sum(self.room["num"])
        self.len_clase = len(self.clase)
        self.len_dia    = len(self.dia)
        self.len_horas   = len(self.horas)
        self.len_cols    = len(self.cols)
        self.len_ROWS    = self.len_clase * self.len_dia * self.len_horas 
             
        self.cromosoma = [i for i in range(self.len_cols)]
        random.shuffle(self.cromosoma)

        self.available_room = np.zeros((self.len_dia * self.len_horas, self.len_room), dtype = np.int8)
        
        shape      = (self.len_clase * self.len_dia * self.len_horas, self.len_cols)
        self.genes = np.zeros(shape, dtype = np.int32)

    def get_cromosoma(self):
        return self.cromosoma
    
    def get_nb_genes(self):
        return self.len_cols
    
    def init_genes(self):
        self.genes[:,:] = 0

    def full_genes(self):
        
        for colm_trab in self.cromosoma:
            (_, room_type, _, units) = self.cols[colm_trab]

            for i in range(self.len_clase):
                start     = i * self.len_dia * self.len_horas
                end       = (i+1) * self.len_dia * self.len_horas - 1
                rand_row  = random.randint(start, end)
                rand_day  = (rand_row % (self.len_dia * self.len_horas)) // self.len_horas
                
                reserv = 0
                for hour in range(self.len_horas):
                    if reserv == units:
                        break
                    tmp_row = start + rand_day * self.len_horas + hour
                    
                    tmp_i = self.room["type"].index(room_type)
                    if tmp_i == -1: raise ValueError("Room no existe")
                    s   = sum(self.room["num"][0:tmp_i])
                    e   = s + self.room["num"][tmp_i] - 1
                    rand_room = random.randint(s, e)
                    
                    if self.genes[tmp_row, colm_trab] == 0 and self.available_room[hour*rand_day, rand_room] == 0:
                        self.available_room[hour*rand_day, rand_room] = 1
                        self.genes[tmp_row, colm_trab] = rand_room + 1 # guarda la clase
                        reserv += 1
                        
                        base  = tmp_row % (self.len_dia * self.len_horas)
                        for i2 in range(self.len_clase):
                            if i != i2:
                                self.genes[base, colm_trab] = -1
                            base += self.len_dia * self.len_horas

                        for col in range(self.len_cols):
                            if col != colm_trab:
                                self.genes[tmp_row, col] = -1

    def entrop(self, clase_nom):

        per_day = np.zeros((self.len_cols, self.len_dia), dtype=np.uint8)

        clase_nom = self.clase.index(clase_nom)
        for day in range(self.len_dia):
            sum_1 = 0
            for hour in range(self.len_horas):
                for row in range(per_day.shape[0]):
                    indx = hour + day*self.len_horas + clase_nom * self.len_dia * self.len_horas
                    if self.genes[indx, row] > 0:
                        per_day[row, day] += 1
        
        entrop_total = 0
        for row in range(per_day.shape[0]):
            entropy = scipy_entropy(per_day[row,:], base=2)
            if entropy == True:
                entrop_total  += entropy
        
        return entrop_total/self.len_cols

    def fitness(self):
        progrm = 0
        (nb_fila, nb_cols) = self.genes.shape
        for i in range(nb_fila):
            for j in range(nb_cols):
                if self.genes[i,j] > 0:
                    progrm+= 1
        
        all_units = 0
        for (_,_,_,u) in self.cols:
            all_units += u

        fitness = progrm/(self.len_clase*all_units)
        
        for i in self.clase:
            fitness -= (self.entrop(i)/self.len_clase)*0.2
        
        return fitness

    def cronograma(self, clase_nom):
        time_table = pd.DataFrame(index=self.horas, columns=self.dia)

        clase_nom = self.clase.index(clase_nom)
        for day in range(self.len_dia):
            for hour in range(self.len_horas):
                for col in range(self.len_cols):
                    indx = hour + day * self.len_horas + clase_nom * self.len_dia * self.len_horas
                    if self.genes[indx, col] > 0:
                        (subject, room_type, lecturer, _) = self.cols[col]
                        
                        i    = self.room["type"].index(room_type)
                        diff = sum(self.room["num"][0:i])
                        room_num = self.genes[indx, col] - 1 - diff
                        
                        time_table.iloc[hour, day] = "{}, {}:{}".format(subject, room_type, room_num)
        return time_table

class poblacion:
    
    def __init__(self, size):
        self.cromosoma = []

        i = 0
        while i < size :
            cromosoma = get_cromosoma(data=data)
            cromosoma.full_genes()
            self.cromosoma.append(cromosoma)
            i += 1
        self.cromosoma.sort(key=lambda x: x.fitness(), reverse=True)

    def get_cromosoma(self):
        return self.cromosoma

    def sort(self, **kargs):
        reverse = kargs.get("revers", False)
        self.cromosoma.sort(key=lambda x: x.fitness(), reverse=reverse)

    def __getitem__(self, indx):
        return self.cromosoma[indx]

    def append(self, cromosoma):
        self.cromosoma.append(cromosoma)
            
    def print_poblacion(self, gen_number):
        print("----------------------- ITERACIONES --------------------------\n")
        print("iteracion ", gen_number, " - aptitud del cromosoma apto:", self.get_cromosoma()[0].fitness())
        print("----------------------------------------------------------------")

class algoritmo_genetico:
    @staticmethod
    def selec_torneo(pop):
        pop_torneo = random.sample(pop.cromosomas(), settings.num_torneo)
        pop_torneo.sort(key=lambda x: x.fitness(), reverse=True)
        
        # return a copy of the parent, otherwise, when the elite
        # is one of the parents, it can be changed with a mutation.
        return copy.deepcopy(pop_torneo[0])

    @staticmethod
    def select(pop):
        partialSum = 0
        sumFitness = 0
        for crom in pop.cromosomas():
            sumFitness += crom.fitness()

        randomShot = random.random() * sumFitness

        i = -1
        while partialSum < randomShot and i < settings.tam_poblacion-1 :
            i += 1
            partialSum += pop[i].fitness()

        return pop[i]

    @staticmethod
    def cruzamientos(parent1, parent2):   
        if random.random() < settings.CROSSING_RATE: 
            child1 = crom(data=settings.data)
            child2 = crom(data=settings.data)

            crossover_index = random.randrange(1, child1.get_nb_genes())
            child_1a = parent1.get_crom()[:crossover_index]
            child_1b = [i for i in parent2.get_crom() if i not in child_1a]
            child1.crom = child_1a + child_1b

            child_2a = parent2.get_crom()[crossover_index:]
            child_2b = [i for i in parent1.get_crom() if i not in child_2a]
            child2.crom = child_2a + child_2b

            child1.fill_genes()
            child2.fill_genes()

            return child1, child2
        else:
            return parent1, parent2

    @staticmethod
    def mutacion(crom):
        if random.random() < settings.var_mutacion:

            random_position1 = random.randrange(0,crom.get_nb_genes())
            random_position2 = random.randrange(0,crom.get_nb_genes())
            
            gene = crom.crom[random_position1]
            crom.crom[random_position1] = crom.crom[random_position2]
            crom.crom[random_position2] = gene

            crom.init_genes()
            crom.fill_genes()
    
    @staticmethod
    def evolve(pop):
        
        new_pop = Population(0)
        LEN_POP = 0

        for i in range(settings.NUMBER_OF_ELITE_cromS):
            new_pop.append(pop[i])
            LEN_POP += 1

        while LEN_POP < settings.tam_poblacion:
            parent1 = algoritmo_genetico.selec_torneo(pop)
            parent2 = algoritmo_genetico.selec_torneo(pop)
            
            child1, child2 = algoritmo_genetico.cruzamientos(parent1, parent2)
            
            algoritmo_genetico.mutacion(child1)
            algoritmo_genetico.mutacion(child2)

            new_pop.append(child1)
            LEN_POP += 1

            if LEN_POP < settings.tam_poblacion:
                new_pop.append(child2)
                LEN_POP += 1

        new_pop.sort(reverse=True)
        return new_pop

if __name__ == "__main__":
    generation_num = 0
    MAX_FITNESS = 1
    poblac = poblacion(tam_poblacion)
    poblacion.print_poblacion(generation_num)

    while poblacion[0].fitness() < MAX_FITNESS and generation_num < settings.MAX_GENERATION_num :
        generation_num += 1
        poblacion = algoritmo_genetico.evolve(poblacion)
        poblacion.print_poblacion(generation_num)
    
    print(poblacion[0].genes)

    print("\n---------- all timetables ------------")
    for c in settings.data["clase"]:
        print("clase : {}".format(c))
        print(poblacion[0].cronograma(c))

        print("---------------------------------------------------\n")
