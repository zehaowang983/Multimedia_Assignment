import numpy as np

def reverse_str(str):
    res = ""
    for _,s in enumerate(str):
        if s=='0':
            res += '1'
        else:
            res += '0'
    return res

def int_size(number):
    str = ""
    if number==0:
        return 0
    if number < 0:
        number = abs(number)
        l = len(bin(number)) - 2
        str = bin(number)[-l:]
        str = reverse_str(str)
    else:
        l = len(bin(number)) - 2
        str = bin(number)[-l:]
    # return (l,str)
    return l 

def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else binstr_flip(binstr)

def binstr_flip(binstr):
    # check if binstr is a binary string
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result

def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    # print(arr)
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i
    # print(last_nonzero)

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []
    # values are binary representations of array elements using SIZE bits
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            # symbols.append((0, 0))
            # values.append(int_to_binstr(0))
            symbols.append(0)
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            # symbols.append((run_length, int_size(elem)))
            symbols.append(symbol2int(run_length,int_size(elem)))
            # values.append(int_to_binstr(elem))
            run_length = 0
    return symbols

def symbol2int(a,b):
    s1 = bin(a)[2:].zfill(4)
    s2 = bin(b)[2:].zfill(4)
    s = s1+s2
    return int(s,2)

a = np.array([32,0,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,1,0,0,0])
# a = np.array([30,0,-1,0,0])
# print(flatten(run_length_encode(a[1:])))
print(run_length_encode(a[1:]))
# print(a)
# print(v)