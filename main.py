import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np


class Vertex:
    def __init__(self, index):
        self.index = index
        self.d = None       # tempo di scoperta
        self.f = None       # tempo di fine
        self.p = None       # predecessore
        self.color = None   # colore di scoperta

    def setColor(self, color):
        if color == "B" or color == 2:              # nodo nero (black)
            self.color = 2
        if color == "G" or color == 1:              # nodo grigio
            self.color = 1
        if color == "W" or color == 0:              # nodo bianco (white)
            self.color = 0
        # else:
        #    print("Color error")

    def getColor(self):
        return self.color


class Graph:
    def __init__(self, num_of_vertex):
        self.m_num_of_vertex = num_of_vertex  # numero di vertici del grafo
        self.m_vertex = []
        for i in range(num_of_vertex):
            self.add_vertex(i)
        self.m_graph = []  # vettore di archi
        self.m_matrix = []  # matrice di adiacenza
        self.time_discovery = None

    # ogni arco di qualsiasi grafo ponderato collega esattamente due vertici e ha un certo peso assegnato
    def set_number_of_vertex(self, num_ver):
        self.m_num_of_vertex = num_ver
        for i in range(self.m_num_of_vertex):
            self.add_vertex(i)

    def add_vertex(self, index):
        self.m_vertex.append(Vertex(index))

    def add_edge(self, vertex1, vertex2, weight):  # aggiunge arco
        self.m_graph.append([vertex1, vertex2, weight])

    def add_matrix(self, matrix):
        self.m_matrix = matrix

    # visita in profondità (Depth First Search)
    def depth_first_search(self):
        print("inizio DFS: ")
        startTimer = timer()
        # final_discoveries = []
        for u in self.m_vertex:         # per ogni vertice appartenente al grafo
            u.setColor("W")
            u.p = None
        # for i in range(self.m_num_of_vertex):
        #     print("Colore Vertice: ", graph.m_vertex[i].color)
        self.time_discovery = 0
        for u in self.m_vertex:
            if u.getColor() == 0:
                self.dfs_visit(u)
        return timer() - startTimer

    def dfs_visit(self, u):
        self.time_discovery += 1
        u.d = self.time_discovery
        print("numero vertice: ", u.index, "inizio discovery: ", u.d)
        u.setColor("G")
        # print("u color gray: ", u.getColor())
        for col in range(self.m_num_of_vertex):
            if self.m_matrix[u.index][col] == 1:    # se nella riga di u della matrice di adiacenza l'arco è uno (esiste)
                v = self.m_vertex[col]              # prende il vertice con indice corrispondente al numero della colonna
                if v.getColor() == 0:
                    v.p = u
                    self.dfs_visit(v)
        u.setColor("B")
        # print("u color black: ", u.getColor())
        self.time_discovery += 1
        u.f = self.time_discovery
        print("numero vertice: ", u.index, "fine discovery: ", u.f)
        # for i in range(self.m_num_of_vertex):
        #    print("Colore Vertice: ", graph.m_vertex[i].color)

    def transpose_graph(self):
        mat_transposed = np.zeros((self.m_num_of_vertex, self.m_num_of_vertex))
        for u in self.m_vertex:         # per ogni vertice appartenente al grafo originale lo reimposta a bianco
            u.setColor("W")
            u.p = None
        self.m_graph[0:len(self.m_graph)] = []      # azzera il vettore degli archi del grafo per sovrascriverlo poi
        for i in range(self.m_num_of_vertex):
            for j in range(self.m_num_of_vertex):
                if self.m_matrix[i][j] == 0:
                    mat_transposed[j][i] = 0
                else:
                    mat_transposed[j][i] = 1
                self.add_edge(i, j, mat_transposed[i][j])
        print("Matrice di adiacenza trasposta: ", mat_transposed)
        print("vettore archi trasposto: ", self.m_graph)
        return self


##########################################################################
#                         GENERATORE GRAFO                               #

# metodo che crea una matrice di adiacenza di un grafo
def adjMatrix_creator(Dim, Prob, graph1):
    matrix = np.zeros((Dim, Dim))
    for i in range(Dim):
        for j in range(Dim):
            # print("riga: ", i, "colonna: ", j)
            if j == i:                              # in un grafo aciclico un vertice non è collegato a se stesso
                matrix[i][j] = 0
            else:
                if random.randint(1, 100) <= Prob and matrix[i][j] == 0:        # se il valore è meno della prob assegnata
                    matrix[i][j] = 1
            graph1.add_edge(i, j, matrix[i][j])         # aggiunge l'arco alla lista di archi
    graph1.m_matrix = matrix
    print("Matrice di adiacenza: ", graph1.m_matrix)


def find_scc(graph_to_scc):
    startTimer = timer()
    graph_to_scc.depth_first_search()
    graph_to_scc.m_vertex = vertex_ord_dec(graph_to_scc)    # ordina il vettore dei vertici del grafo in ordine di fine scoperta decrescente
    graph_transposed = graph_to_scc.transpose_graph()
    graph_transposed.depth_first_search()       # con ordine di visita decrescente rispetto alle u.f calcolate nella prima dfs
    return timer() - startTimer


def vertex_ord_dec(G):
    G.m_vertex.sort(key=lambda x: x.f, reverse=True)
    # for i in range(0, G.m_num_of_vertex):
    #     print("vettore vertici dec: ", G.m_vertex[i].f)
    return G.m_vertex


def test_num_vertex_up1():
    totTimeL = []
    totTimeH = []
    avgTimeL = []
    avgTimeH = []
    dim = 5
    d = []

    while dim <= 25:
        d.append(dim)
        # graphL.set_number_of_vertex(dim)
        # graphH.set_number_of_vertex(dim)
        probL = 30
        probH = 70
        for i in range(0, 5):
            graphL = Graph(dim)
            graphH = Graph(dim)
            adjMatrix_creator(dim, probL, graphL)
            adjMatrix_creator(dim, probH, graphH)
            # print("tempo dfs: ", graph.depth_first_search())
            # print("lista di archi: ", graph.m_graph)
            totTimeL.append(find_scc(graphL))
            totTimeH.append(find_scc(graphH))
            print("tempo scc: ", find_scc(graphL))
            print("tempo scc: ", find_scc(graphH))
        sumTL = 0
        sumTH = 0
        for k in range(0, len(totTimeL)):
            sumTL += totTimeL[k]
            sumTH += totTimeH[k]
        mediaTL = sumTL / len(totTimeL)
        mediaTH = sumTH / len(totTimeH)
        avgTimeL.append(mediaTL)
        avgTimeH.append(mediaTH)
        print("Media tempo dim crescente e bassa densità: ", avgTimeL)
        print("Media tempo dim crescente e alta densità: ", avgTimeH)
        dim += 5
    plt.plot(d, avgTimeL, label="SCC per grafo a bassa densità di archi")
    plt.plot(d, avgTimeH, label="SCC per grafo a alta densità di archi")
    plt.legend()
    plt.ylabel('Secondi')
    plt.xlabel('Numero vertici')
    plt.show()


def test_num_vertex_up2():
    totTimeL = []
    totTimeH = []
    avgTimeL = []
    avgTimeH = []
    dim = 5
    d = []

    while dim <= 25:
        d.append(dim)
        # graphL.set_number_of_vertex(dim)
        # graphH.set_number_of_vertex(dim)
        probL = 10
        probH = 90
        for i in range(0, 5):
            graphL = Graph(dim)
            graphH = Graph(dim)
            adjMatrix_creator(dim, probL, graphL)
            adjMatrix_creator(dim, probH, graphH)
            # print("tempo dfs: ", graph.depth_first_search())
            # print("lista di archi: ", graph.m_graph)
            totTimeL.append(find_scc(graphL))
            totTimeH.append(find_scc(graphH))
            print("tempo scc: ", find_scc(graphL))
            print("tempo scc: ", find_scc(graphH))
        sumTL = 0
        sumTH = 0
        for k in range(0, len(totTimeL)):
            sumTL += totTimeL[k]
            sumTH += totTimeH[k]
        mediaTL = sumTL / len(totTimeL)
        mediaTH = sumTH / len(totTimeH)
        avgTimeL.append(mediaTL)
        avgTimeH.append(mediaTH)
        print("Media tempo dim crescente e bassa densità: ", avgTimeL)
        print("Media tempo dim crescente e alta densità: ", avgTimeH)
        dim += 5
    plt.plot(d, avgTimeL, label="SCC per grafo a bassa densità di archi")
    plt.plot(d, avgTimeH, label="SCC per grafo a alta densità di archi")
    plt.legend()
    plt.ylabel('Secondi')
    plt.xlabel('Numero vertici')
    plt.show()


def test_density_up():
    totTimeL = []
    totTimeH = []
    avgTimeL = []
    avgTimeH = []
    dens = 10
    d = []
    dimL = 5
    dimH = 20
    while dens <= 90:
        d.append(dens)
        for i in range(0, 5):
            graphL = Graph(dimL)
            graphH = Graph(dimH)
            adjMatrix_creator(dimL, dens, graphL)
            adjMatrix_creator(dimH, dens, graphH)
            # print("tempo dfs: ", graph.depth_first_search())
            # print("lista di archi: ", graph.m_graph)
            totTimeL.append(find_scc(graphL))
            totTimeH.append(find_scc(graphH))
            print("tempo scc: ", find_scc(graphL))
            print("tempo scc: ", find_scc(graphH))
        sumTL = 0
        sumTH = 0
        for k in range(0, len(totTimeL)):
            sumTL += totTimeL[k]
            sumTH += totTimeH[k]
        mediaTL = sumTL / len(totTimeL)
        mediaTH = sumTH / len(totTimeH)
        avgTimeL.append(mediaTL)
        avgTimeH.append(mediaTH)
        print("Media tempo densità crescente e bassa dimensione: ", avgTimeL)
        print("Media tempo densità crescente e alta dimensione: ", avgTimeH)
        dens += 20
    plt.plot(d, avgTimeL, label="SCC per grafo con basso numero di nodi")
    plt.plot(d, avgTimeH, label="SCC per grafo con alto numero di nodi")
    plt.legend()
    plt.ylabel('Secondi')
    plt.xlabel('Fattore di densità')
    plt.show()


if __name__ == '__main__':
    print("Test 1 dimensione crescente: ")
    test_num_vertex_up1()
    print("Test 2 dimensione crescente: ")
    test_num_vertex_up2()
    print("Test 3 densità crescente: ")
    test_density_up()
