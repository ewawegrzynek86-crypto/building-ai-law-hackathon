import math

def sigmoid(x):
    """Funkcja aktywacji Sigmoid"""
    return 1 / (1 + math.exp(-x))

def calculate_node_a1():
    # Dane wejściowe (Input nodes)
    inputs = [0.5, 0.5, 1.0]
    
    # Wagi dla węzła a1 (Weights)
    weights = [0.1, 0.4, 0.2]
    
    # Obciążenie (Bias)
    bias = 3.348
    
    # 1. Obliczanie kombinacji liniowej (zgodnie z naszym zadaniem)
    # linear_combination = (1.0 * 0.1) + (1.0 * 0.4) + (0.0 * 0.2) + 3.348
    linear_combination = sum(i * w for i, w in zip(inputs, weights)) + bias
    
    # 2. Obliczanie wyjścia z funkcją Sigmoid
    output_sigmoid = sigmoid(linear_combination)
    
    # 3. Obliczanie wyjścia z funkcją Identity (f(x) = x)
    output_identity = linear_combination
    
    print(f"--- Wyniki dla węzła a1 ---")
    print(f"Kombinacja liniowa: {linear_combination}")
    print(f"Output (Sigmoid): {output_sigmoid:.4f}")
    print(f"Output (Identity): {output_identity}")

if __name__ == "__main__":
    calculate_node_a1()
