import numpy as np

def dcpm(Zs): 
    dcs = []
    for i in range(3):
        # dcs.append(Zs[i][:][0])
        dc = []
        blocks = len(Zs[i])
        for j in range(blocks):
            if j==0:
                dc.append(int_size(Zs[i][j][0]))
            else:
                dc.append(int_size(Zs[i][j][0]-Zs[i][j-1][0]))
        dcs.append(dc)
    return dcs

def dc(Zs):
    dcs = []
    for i in range(3):
        # dcs.append(Zs[i][:][0])
        dc = []
        blocks = len(Zs[i])
        for j in range(blocks):
            if j==0:
                dc.append(Zs[i][j][0])
            else:
                dc.append(Zs[i][j][0]-Zs[i][j-1][0])
        dcs.append(dc)
    return dcs

def rlc(Zs):
    acs = []
    for i in range(3):
        # dcs.append(Zs[i][:][0])
        ac = []
        blocks = len(Zs[i])
        for j in range(blocks):
            ac.append(run_length_encode(Zs[i][j][1:]))
        acs.append(ac)
    return acs

def ac(Zs):
    acs = []
    for i in range(3):
        # dcs.append(Zs[i][:][0])
        ac = []
        blocks = len(Zs[i])
        for j in range(blocks):
            # ac.append(run_length_encode(Zs[i][j][1:]))
            ac.append(Zs[i][j][1:])
        acs.append(ac)
    return acs

def rlc_values(Zs):
    valuess = []
    for i in range(3):
        values = []
        blocks = len(Zs[i])
        for j in range(blocks):
            arr = Zs[i][j][1:]
            value = []
            # determine where the sequence is ending prematurely
            last_nonzero = -1
            # print(arr)
            for k, elem in enumerate(arr):
                if elem != 0:
                    last_nonzero = k
            # print(last_nonzero)
            run_length = 0

            for k, elem in enumerate(arr):
                if k > last_nonzero:
                    value.append(int_to_binstr(0))
                    break
                elif elem == 0 and run_length < 15:
                    run_length += 1
                else:
                    value.append(int_to_binstr(elem))
                    run_length = 0
            
            values.append(value)

        valuess.append(values)
    return valuess

def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])

def zigzag_points(rows, cols):
    # constants for directions
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # move the point in different directions
    def move(direction, point):
        return {
            UP: lambda point: (point[0] - 1, point[1]),
            DOWN: lambda point: (point[0] + 1, point[1]),
            LEFT: lambda point: (point[0], point[1] - 1),
            RIGHT: lambda point: (point[0], point[1] + 1),
            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
        }[direction](point)

    # return true if point is inside the block bounds
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # start in the top-left cell
    point = (0, 0)

    # True when moving up-right, False when moving down-left
    move_up = True

    for i in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)

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

def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def binstr_flip(binstr):
    # check if binstr is a binary string
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else binstr_flip(binstr)


def flatten(lst):
    return [item for sublist in lst for item in sublist]

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

    # values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            # symbols.append(0)
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            # size = bits_required(elem)
            symbols.append((run_length, int_size(elem)))
            # symbols.append(symbol2int(run_length,int_size(elem)))
            run_length = 0
    return symbols

def symbol2int(a,b):
    s1 = bin(a)[2:].zfill(4)
    s2 = bin(b)[2:].zfill(4)
    s = s1+s2
    return int(s,2)

def write_to_file(filepath, dcs,dcs_values, acs, acs_bin,tables):
    count = 0
    try:
        f = open(filepath, 'wb')
    except FileNotFoundError as e:
        raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e

    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # 16 bits for 'table_size'
        f.write(bytes(uint_to_binstr(len(tables[table_name]), 16), encoding = "utf8"))
        # f.write(len(tables[table_name],16))
        count += 16

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                # 4 bits for the 'category'
                # 4 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(bytes(uint_to_binstr(key, 4), encoding = "utf8"))
                f.write(bytes(uint_to_binstr(len(value), 4), encoding = "utf8"))
                f.write(bytes(value, encoding = "utf8"))
                count += 8
                count +=len(bytes(value, encoding = "utf8"))
                # f.write(key, 4)
                # f.write(len(value), 4)
                # f.write(value)
            else:
                # 4 bits for 'run_length'
                # 4 bits for 'size'
                # 8 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                # print(key)
                f.write(bytes(uint_to_binstr(key[0], 4), encoding = "utf8"))
                f.write(bytes(uint_to_binstr(key[1], 4), encoding = "utf8"))
                f.write(bytes(uint_to_binstr(len(value), 8), encoding = "utf8"))
                f.write(bytes(value, encoding = "utf8"))
                count += 16
                count +=len(bytes(value, encoding = "utf8"))
                # f.write(key[0], 4)
                # f.write(key[1], 4)
                # f.write(len(value), 8)
                # f.write(value)


    # 32 bits for 'Y_blocks_count'
    Y_blocks_count = len(dcs[0])
    f.write(bytes(uint_to_binstr(Y_blocks_count, 32), encoding = "utf8"))
    count += 32

    # for Y channel 
    for i in range(Y_blocks_count):
        category = int_size(dcs_values[0][i])
        # symbols = run_length_encode(acs[0][i])
        symbols = acs[0][i]

        dc_table = tables['dc_y'] 
        ac_table = tables['ac_y']

        f.write(bytes(dc_table[category], encoding = "utf8"))
        f.write(bytes(int_to_binstr(dcs_values[0][i]), encoding = "utf8"))
        count += len(bytes(dc_table[category], encoding = "utf8"))
        count += len(bytes(int_to_binstr(dcs_values[0][i]), encoding = "utf8"))

        for j in range(len(symbols)):
            f.write(bytes(ac_table[tuple(symbols[j])], encoding = "utf8"))
            f.write(bytes(acs_bin[0][i][j], encoding = "utf8"))
            count += len(bytes(ac_table[tuple(symbols[j])], encoding = "utf8"))
            count += len(bytes(acs_bin[0][i][j], encoding = "utf8"))
    
    # 32 bits for 'C_blocks_count'
    C_blocks_count = len(dcs[1])
    f.write(bytes(uint_to_binstr(C_blocks_count, 32), encoding = "utf8"))
    count += 32

    # for Cr channel
    for i in range(C_blocks_count):
        category = int_size(dcs_values[1][i])
        # symbols = run_length_encode(acs[1][i])
        symbols = acs[1][i]

        dc_table = tables['dc_c'] 
        ac_table = tables['ac_c']

        f.write(bytes(dc_table[category], encoding = "utf8"))
        f.write(bytes(int_to_binstr(dcs_values[1][i]), encoding = "utf8"))
        count += len(bytes(dc_table[category], encoding = "utf8"))
        count += len(bytes(int_to_binstr(dcs_values[1][i]), encoding = "utf8"))

        for j in range(len(symbols)):
            f.write(bytes(ac_table[tuple(symbols[j])], encoding = "utf8"))
            f.write(bytes(acs_bin[1][i][j], encoding = "utf8"))
            count += len(bytes(ac_table[tuple(symbols[j])], encoding = "utf8"))
            count += len(bytes(acs_bin[1][i][j], encoding = "utf8"))
    
    # for Cb channel
    for i in range(C_blocks_count):
        category = int_size(dcs_values[2][i])
        # symbols = run_length_encode(acs[2][i])
        symbols = acs[2][i]

        dc_table = tables['dc_c'] 
        ac_table = tables['ac_c']

        f.write(bytes(dc_table[category], encoding = "utf8"))
        f.write(bytes(int_to_binstr(dcs_values[2][i]), encoding = "utf8"))     
        count += len(bytes(dc_table[category], encoding = "utf8"))
        count += len(bytes(int_to_binstr(dcs_values[2][i]), encoding = "utf8"))

        for j in range(len(symbols)):
            f.write(bytes(ac_table[tuple(symbols[j])], encoding = "utf8"))
            f.write(bytes(acs_bin[2][i][j], encoding = "utf8"))
            count += len(bytes(ac_table[tuple(symbols[j])], encoding = "utf8"))
            count += len(bytes(acs_bin[2][i][j], encoding = "utf8"))
    f.close()
    return count

