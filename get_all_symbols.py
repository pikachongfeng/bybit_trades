import os

from utils.get_bybit_data import *

if __name__ == "__main__":
    
    symbols = get_symbols()
    new_symbols = []
    for symbol in symbols:
        flag = True
        for file in os.listdir('./'):
            if symbol in file:
                flag = False
        if flag:
            new_symbols.append(symbol)

    with open('./selected_symbols.txt','w') as f:
        for symbol in new_symbols:
            f.write(symbol)
            f.write('\n')