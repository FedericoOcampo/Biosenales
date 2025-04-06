import numpy as np
import matplotlib.pyplot as plt

#a vectores
a = np.array([3.1, 1, -0.5, -3.2, 6])
b = np.array([1, 3, 2.2, 5.1, 1])

#b multi escalar
# R:// se debe implementar un arreglo de numpy para poder realizar la multiplicación
# por medio de funciones ya establecidas y se debe cumplir la condición de dimensión
Multi_escalar = np.dot(a,b)
print('\n',Multi_escalar)

#c multi punto a punto
Multi_punto = a*b
print('\n',Multi_punto)

#d matriz
A=np.array([[2, -1, -3], 
           [4, 1.5, -2.5], 
           [7.3, -0.9, 0.2]])
print('\n''A: \n',A)

#e transpuesta
Trans = A.T
print('\n',Trans)

#f ones
matriz_unos = np.ones([3,3])
print('\n',matriz_unos)

# round (redondea un numero definido de decimales)
matriz_round = np.round(A,2)
print('\n',A)

# ceil (redondea un numero hacia arriba)
matriz_ceil = np.ceil(A)
print('\n', A)

#floor (redondea un numero abajo)
matriz_floor = np.floor(A)
print('\n', A)

#g 1er fila 3er columna
valor = A[0,2]
print('\n',valor)

#h 2da fila
segunda_fila = A[1,:] 
print('\n',segunda_fila)

#i shape
dimension = A.shape
print('\n', dimension)

#j
n=np.arange(0,101) #(empieza, termina)
y = np.sin(np.pi*0.12*n)

#k
y2 = np.cos(np.pi*2*0.03*n)

#i
s = y+y2
t = y*y2

#m grafica Y y Y2

plt.figure()
plt.plot(n, y, label='y[n] = sin(π * 0.12 * n)', color='blue')
plt.plot(n, y2, label='y2[n] = cos(2π * 0.03n)', color = 'green')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

#n grafica de s y t
plt.figure()
plt.plot(n, s, label='s[n] = y[n] + y2[n]', color='blue')
plt.plot(n, t, label='t[n] = y[n] * y2[n]', color = 'green')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

#---------------------------------------------------------------
# %%

import pandas as pd
import numpy as np

def organiza_notas(notas):

    serie_notas = pd.Series(notas)

    max = serie_notas.max()
    min = serie_notas.min()
    media = serie_notas.mean()
    desv = serie_notas.std()
  
    return pd.Series([max, min, media, desv ], index=['nota max', 'nota min', 'media', 'desviación'])


notas_obtenidad = {'Juan': 1, 'Camilo': 2, 'Luisa': 4.5, 'Miguel': 4.5, 'Juliana': 2}

print(organiza_notas(notas_obtenidad))

# %%
import pandas as pd
import numpy as np

bd = pd.read_csv('datos.csv', sep=';') #se separa por ;
bd.shape
print(bd.shape)

#b mostrar primeras y ultimas filas
print(bd.head()) #muestra los primeros 5 datos
print(bd.tail()) #muestra los ultimos 5 datos

#c borrar columna
bd.drop('Unnamed: 0',axis=1, inplace=True) #axis=1 es para borrar columnas y axis=0 para borrar filas 

#d calcular imc en cada fila y agregarlo como columna
bd['IMC'] = bd['Height']/(bd['Weight']/100)**2
print('\n',bd)

#e clasificar por IMC

for i in range(len(bd)):
    if bd['IMC'][i] < 18.5:
        bd.loc[i, 'Clasificacion'] = 'Bajo peso'
    elif bd['IMC'][i] >= 18.5 and bd['IMC'][i] < 24.9:
        bd.loc[i, 'Clasificacion'] = 'Normal'
    elif bd['IMC'][i] >= 25 and bd['IMC'][i] < 29.9:
        bd.loc[i, 'Clasificacion'] = 'Sobrepeso'
    else:
        bd.loc[i, 'Clasificacion'] = 'Obesidad'

        
print('\n',bd)





