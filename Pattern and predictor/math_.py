import numpy as np

'''
This file contains the functions used to calculate points of a line
author: @jsotoL
created: 01-03-2023
version: 1.0
'''

def ecuación_de_la_recta(x1, y1, x2, y2):
    """Calcula la ecuación de la recta que pasa por los puntos (x1, y1) y (x2, y2)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def calcular_puntos_de_la_recta(m, b, x1, x2):
    """Calcula los puntos de la recta que pasa por los puntos (x1, y1) y (x2, y2)."""
    y1 = m * x1 + b
    y2 = m * x2 + b
    return y2

def logistic_growth(x, a, b, c):
    """Función de crecimiento logístico."""
    return c / (1 + np.exp(-(x - b) / a)) #a = velocidad de crecimiento, b = punto medio, c = valor máximo x = tiempo

def agregar_a_lista(lista, x1):
    """Agrega los puntos de la recta a la lista."""
    lista.append(x1)
    return lista

def diff_in_percentages(x1,x2):
    return ((x2-x1)/x1)*100

def knapsack(values, weights, max_weight, max_overload):
    n = len(values)
    dp = [[0] * (max_weight + max_overload + 1) for _ in range(n + 1)]
    item = [[0] * (max_weight + max_overload + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(max_weight + max_overload + 1):
            if weights[i - 1] <= j:
                if dp[i - 1][j] > dp[i - 1][j - weights[i - 1]] + values[i - 1]:
                    dp[i][j] = dp[i - 1][j]
                    item[i][j] = 0
                else:
                    dp[i][j] = dp[i - 1][j - weights[i - 1]] + values[i - 1]
                    item[i][j] = 1
            else:
                dp[i][j] = dp[i - 1][j]
                item[i][j] = 0
    max_value = 0
    last_index = 0
    for j in range(max_weight, max_weight + max_overload + 1):
        if dp[n][j] > max_value:
            max_value = dp[n][j]
            last_index = j
    result = []
    i = n
    j = last_index
    while i > 0 and j > 0:
        if item[i][j] == 1:
            result.append(i - 1)
            j -= weights[i - 1]
        i -= 1
    result.reverse()
    return max_value, result