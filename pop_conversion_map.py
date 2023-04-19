### Piano
PIANO = list(range(6))
CHROMATIC_PERCUSSION = [10, 8, 9, 11, 12, 13, 14, 15]

### Electric Keyboard
SYNTHESIZER = [87, 80, 81, 82, 83, 84, 85, 86] + list(range(88, 96)) + list(range(96, 104))
ORGAN = [19, 16, 17, 18, 20, 21, 22, 23]

### Guitar - all plucking instruments
GUITAR = [24, 25, 45, 46, 6, 7, 104, 105, 106, 107] # 46 is harp, 45 is pizzicato, 6, 7 are harpsichord

### Electric Guitar
ELECTRIC_GUITAR = [27, 29, 26, 28, 30, 31]


### Bass
BASS = [34, 32, 33, 35, 36, 37, 38, 39]

### Human Voice - strings, wind and choir
STRINGS = [40, 41, 42, 43, 44, 48, 49, 50, 51, 110]

TRUMPET = [60, 56, 57, 59]
BRASS_SECTION = [61, 62, 63]
SAX = [64, 65, 66, 67]
OBOE = [68, 69]
PIPE = [73, 72, 74, 75, 76, 77, 78, 79]
WIND = TRUMPET + [58, 60] + BRASS_SECTION + SAX + OBOE + PIPE + [70, 71, 72, 73]

CHOIR = [52, 53, 54]

HUMAN_VOICE = STRINGS + WIND + CHOIR

ALL_CLASSES = [PIANO, CHROMATIC_PERCUSSION, SYNTHESIZER, ORGAN, GUITAR, ELECTRIC_GUITAR, BASS, HUMAN_VOICE]
inst_class_dict = {c[0]: c for c in ALL_CLASSES}

def reverse_dict(d: dict) -> dict:
    res_dict = {}
    for k in d:
        for v in d[k]:
            res_dict[v] = k
    return res_dict


conversion_map = reverse_dict(inst_class_dict)
