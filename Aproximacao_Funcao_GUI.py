from re import sub
from tkinter.simpledialog import SimpleDialog
import PySimpleGUI as sg
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

def aproximacao_grafico(a,b,c,d,e):
    entradas=1
    num_neuronios=int(a)
    taxa_aprendizado=float(b)
    erro_tolerado=0.001
    listaciclo=[]#armazena a quantidade de ciclos
    listaerro=[]#armazena a quantidade de erros em cada ciclo
    xmin=-int(c)
    xmax=int(d)
    npontos=int(e)

    #Gerando o arquivo de entradas
    x1=np.linspace(xmin,xmax,npontos)
    #Criando as matrizes de entradas
    x=np.zeros((npontos,1))
    for i in range(npontos):
        x[i][0]=x1[i]
    (amostras,vsai)=np.shape(x)

    t1=(np.cos(x))*(np.cos(2*x))
    t=np.zeros((1,amostras))
    for i in range(amostras):
        t[0][i]=t1[i]
    (vsai, amostras)=np.shape(t)
#=================================================================
    #Gerando os pesos sinápticos aleatoriamente
    #Deve criar os pesos e bias entre a camada de entrada e intermediária   
    vanterior=np.zeros((entradas,num_neuronios))
    aleatorio=0.5
    #cria valores aleatorios entre +0.5 e -0.5
    for i in range(entradas):
        for j in range(num_neuronios):
            vanterior[i][j]=rd.uniform(-aleatorio,aleatorio)
    v0anterior=np.zeros((1,num_neuronios))
    #gera os bias na camada intermediaria
    
    #preencher os valores
    for j in range(num_neuronios):
        v0anterior[0][j]=rd.uniform(-aleatorio,aleatorio)
        #deve sempre começar pela linha 0 
#=====================================================================   
     #Deve criar os pesos e bias da camada de intermediaria e saida 
    wanterior = np.zeros((num_neuronios,vsai))
    aleatorio=0.2
    for i in range(num_neuronios):
        for j in range(vsai):
            wanterior[i][j]=rd.uniform(-aleatorio,aleatorio)
    w0anterior=np.zeros((1,vsai))
    for j in range(vsai):
        w0anterior[0][j]=rd.uniform(-aleatorio, aleatorio)
     
    #Matrizes de atualização de pesos e valores de saída
    #Todas as matrizes vão ser iniciadas com zeros
    #Essas variaveis vão armazenar os valores das novas e antigas variaveis
    vnovo=np.zeros((entradas,num_neuronios))
    v0novo=np.zeros((1,num_neuronios))
    wnovo=np.zeros((num_neuronios,vsai))
    w0novo=np.zeros((1,vsai))
    
    #Valores que chegarão na camada intermediária  
    zin=np.zeros((1,num_neuronios))
    z=np.zeros((1,num_neuronios))#Funcao de ativacao
    
    #Deve ter os deltas para realizar as atualizacoes dos pesos
    deltinhak=np.zeros((vsai,1))
    deltaw0=np.zeros((vsai,1))
    deltinha=np.zeros((1,num_neuronios))
    
    xaux=np.zeros((1,entradas))
    h=np.zeros((vsai,1))
    target=np.zeros((vsai,1))
    deltinha2=np.zeros((num_neuronios,1))
    #Necessario para fazer a transposicao de linha para coluna
    
    ciclo=0
    errototal=100000#deve colocar esse valor para entrar no while
    
    #Treinar a rede ate chegar no erro tolerado
    while erro_tolerado<errototal and ciclo<=5000:
        errototal=0
        #Ao entrar  o errototal eh zerado e
        #vai ser calculado ciclo a ciclo
        for padrao in range(amostras):
            for j in range(num_neuronios):
            #laco que equivale a quantidade de neuronios
                zin[0][j]=np.dot(x[padrao,:],vanterior[:,j])+v0anterior[0][j]
                #calculo para os valores que chegam na camada intermediaria, 
                #multiplicacao de matrizes e soma do bias para o nó da 
                #camada intermediária

            z=np.tanh(zin)
            #O z vai submeter os valores(da camada intermediaria) 
            #na funcao de ativacao  
            #Nesse caso vai usar a tangente hiperbólica
            
            yin=np.dot(z,wanterior)+w0anterior
            #Formula do backpropagation para os valores que 
            #chegam a camada de saida
            
            y=np.tanh(yin)
            #Aplica a funcao de ativacao em yin
            
            #Dessa forma obtem todos os valores da camada de 
            #saida
            
            #O h precisa mudar de linha para coluna, sendo necessario
            #fazer uma transposicao
            for m in range(vsai):
                h[m][0]=y[0][m]
                #transposta de h
            for m in range(vsai):
                target[m][0]=t[0][padrao]
                #armazena a variavel target, se o padrao for de
                #ordem 0 ele vai na coluna 0 de t
                
            errototal=errototal+np.sum(0.5*((target-h)**2))
            #Aplicando a formula do erro do algoritmo backpropagation
            
            
            #Aplicar as formulas para atualiazar pesos entre a 
            #camada intermediária e de saídas
            deltinhak=(target-h)*(1+h)*(1-h)
            #Formula da retropopagação do erro
            deltaw=taxa_aprendizado*(np.dot(deltinhak,z))
            deltaw0=taxa_aprendizado*deltinhak
            deltinhain=np.dot(np.transpose(deltinhak),np.transpose(wanterior))
            
            deltinha=deltinhain*(1+z)*(1-z)
            #Composição da derivada
            
            #transposicao de matrizes de linha para coluna
            for m in range(num_neuronios):
                deltinha2[m][0]=deltinha[0][m]
            #transposicao do vetor de entradas x
            for k in range(entradas):
                xaux[0][k]=x[padrao][k]
                
            #Aplicando as formulas para as matrizes de 
            #atualizacao de peso
            deltav=taxa_aprendizado*np.dot(deltinha2,xaux)
            deltav0=taxa_aprendizado*deltinha
            
            #Realizando atualização de pesos
            vnovo=vanterior+np.transpose(deltav)
            v0novo=v0anterior+np.transpose(deltav0)
            #v0novo=vanterior+deltav0
            wnovo=wanterior+np.transpose(deltaw)
            w0novo=w0anterior+np.transpose(deltaw0)
            vanterior=vnovo
            v0anterior=v0novo
            wanterior=wnovo
            w0anterior=w0novo
        ciclo=ciclo+1
        listaciclo.append(ciclo)#adiciona o resultado de ciclo na listaciclo
        listaerro.append(errototal)
        print("CICLO\t ERRO")   
        print(ciclo,'\t',errototal)

        zin2=np.zeros((1,num_neuronios))
        z2=np.zeros((1,num_neuronios))
        t2=np.zeros((amostras,1))
        #Calculo das operações
        for i in range(amostras):
            for j in range(num_neuronios):
                zin2[0][j]=np.dot(x[i,:],vanterior[:,j])+v0anterior[0][j]
                #Formula do z inicial
                z2=np.tanh(zin2)
                #Apllicando a funcao de ativacao
                #tangente hiperbolica em zin
                
            yin2=np.dot(z2,wanterior)+w0anterior
            y2=np.tanh(yin2)
            
            t2[i][0]=y2

        update_graph(x,t1,t2)
        update_error(listaciclo,listaerro)


def update_graph(x, t1, t2):

    axes = fig.axes
    axes[0].cla()
    axes[0].plot(x,t1,color="red")
    axes[0].plot(x,t2,color="blue")
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack()

def update_error(ciclo, erro):

    axes2 = fig2.axes
    axes2[0].cla()
    axes2[0].set_xlim([0, ciclo[-1]*2 if ciclo[-1]*2 < 10000 else 10000])
    axes2[0].set_xlabel('Ciclos')
    axes2[0].set_ylabel('Erro')
    axes2[0].plot(ciclo,erro ,color="purple")
    figure_canvas_agg2.draw()
    figure_canvas_agg2.get_tk_widget().pack()
    window.refresh()

col1 = [
            [sg.Text('Aproximação de Função', font=('bold', 20), justification='center')],
            [sg.Text('Quantidade de neurônios: ', font=('normal',16), size=(30,1)), sg.InputText()],
            [sg.Text('Taxa de Aprendizado: ', font=('normal',16), size=(30,1)), sg.InputText()],
            [sg.Text('Valor mínimo do gráfico: ', font=('normal',16), size=(30,1)), sg.InputText()],
            [sg.Text('Valor máximo do gráfico: ', font=('normal',16), size=(30,1)), sg.InputText()],
            [sg.Text('Número de pontos do gráfico: ', font=('normal',16), size=(30,1)), sg.InputText()],
            [sg.ReadButton('Treinar')]
       ]

col2 = [
            [sg.Canvas(size=(640,480),key='-CANVAS-')],
            [sg.Canvas(size=(640,480),key='-CANVAS-ERROR-')]
       ]

layout = [
            [sg.Column(col1),sg.Column(col2)]
         ]

window = sg.Window('Aproximação de Função', layout, finalize=True)

fig = Figure(figsize=(6.4,4.8))
fig.add_subplot(111).plot([],[])
figure_canvas_agg = FigureCanvasTkAgg(fig,window['-CANVAS-'].tk_canvas)
figure_canvas_agg.draw()
figure_canvas_agg.get_tk_widget().pack()

fig2 = Figure(figsize=(6.4,4.8))
a2 = fig2.add_subplot(111)
a2.set_xlabel('Ciclos')
a2.set_ylabel('Erro')
a2.plot([],[])
figure_canvas_agg2 = FigureCanvasTkAgg(fig2,window['-CANVAS-ERROR-'].tk_canvas)
figure_canvas_agg2.draw()
figure_canvas_agg2.get_tk_widget().pack()

while True:
    event, values = window.read()
    if event == 'Cancel': 
        break
    if event == 'Treinar':
        aproximacao_grafico(values[0],values[1],values[2],values[3],values[4])