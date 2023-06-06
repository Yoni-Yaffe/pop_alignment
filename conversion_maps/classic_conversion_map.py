### Piano
PIANO = list(range(6)) + [7]
ORGAN = [19, 16, 17, 18, 20, 21, 22, 23]

### Guitar
GUITAR = list(range(24, 40)) + [6, 46, 45]


### STRINGS
STRINGS = [40, 41, 42, 43, 48, 49, 105]

### WINDS
TRUMPET = [60, 56, 57, 59]
BRASS_SECTION = [61, 62, 63]
SAX = [64, 65, 66, 67]
OBOE = [68, 69]
PIPE = [73, 72, 74, 75, 76, 77, 78, 79]
WIND = TRUMPET + [58, 60] + BRASS_SECTION + SAX + OBOE + PIPE + [70, 71, 72, 73] + [52]


ALL_CLASSES = [PIANO + ORGAN, GUITAR, STRINGS,  WIND, [47, 11]]
inst_class_dict = {c[0]: c for c in ALL_CLASSES}

def reverse_dict(d: dict) -> dict:
    res_dict = {}
    for k in d:
        for v in d[k]:
            res_dict[v] = k
    return res_dict


conversion_map = reverse_dict(inst_class_dict)
