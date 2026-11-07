
# 改成自己的
SVG_CATEGORIES = [
    # 1-6 doors
    {"color": [224, 62, 155], "isthing": 1, "id": 0, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 1, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 2, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 3, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 4, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 5, "name": "rolling door"},
    # 7-10 window
    {"color": [96, 78, 245], "isthing": 1, "id": 6, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 7, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 8, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 9, "name": "opening symbol"},
    # 11-27: furniture
    {"color": [122, 181, 145], "isthing": 1, "id": 10, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 11, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 12, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 13, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 14, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 15, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 16, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 17, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 18, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 19, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 20, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 21, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 22, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 23, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 24, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 25, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 26, "name": "toilet"},
    # 28:stairs
    {"color": [247, 206, 75], "isthing": 1, "id": 27, "name": "stairs"},
    # 29-30: equipment
    {"color": [237, 112, 45], "isthing": 1, "id": 28, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 29, "name": "escalator"},

    # 31-35: uncountable
    {"color": [172, 107, 151], "isthing": 0, "id": 30, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 31, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 32, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 33, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 34, "name": "railing"},
    {"color": [0, 0, 0], "isthing": 0, "id": 35, "name": "bg"},    
]



color_pallete = {
    35: (0, 0, 0),
    0: (241, 226, 226), 1: (214, 168, 168), 2: (187, 111, 111),  # door
    3: (187, 111, 111), 4: (87, 41, 41), 5: (29, 14, 14),  # door
    6: (226, 232, 241), 7: (168, 186, 214), 8: (111, 140, 187), 9: (68, 97, 144),  # window
    27: (244, 247, 50),  # stairs
    17: (247, 214, 253), 19: (231, 132, 250), 23: (103, 5, 123),  # home appliance
    10: (245, 250, 245), 11: (224, 241, 225), 12: (203, 232, 204), 13: (182, 223, 184),  # furniture
    14: (161, 214, 164), 15: (140, 205, 143), 16: (119, 196, 123), 18: (98, 187, 103),  # furniture
    20: (78, 177, 83), 21: (68, 157, 73), 22: (59, 136, 63), 24: (50, 115, 53),
    25: (41, 94, 44), 26: (32, 73, 34),  # furniture
    28: (239, 220, 220), 29: (177, 78, 78),  # equipment
    30: (222, 210, 179), 31: (200, 180, 128), 32: (177, 150, 78), 33: (127, 107, 55), 34: (76, 64, 33),  # stuff
}

# color_pallete = {
#     0: (245, 250, 245),
#     1: (241, 226, 226), 2: (214, 168, 168), 3: (187, 111, 111),
#     4: (187, 111, 111), 5: (87, 41, 41), 6: (29, 14, 14),
#     7: (226, 232, 241), 8: (168, 186, 214), 9: (111, 140, 187), 10: (68, 97, 144),
#     11: (127, 127, 127),
# }

class AnnoList:
    def __init__(self):
        # all classes
        self.anno_list_all = {
            'single door': 0,
            'double door': 1,
            'sliding door': 2,
            'folding door': 3,
            'revolving door': 4,
            'rolling door': 5,

            'window': 6,
            'bay window': 7,
            'blind window': 8,
            'opening symbol': 9,

            'sofa': 10,
            'bed': 11,
            'chair': 12,
            'table': 13,
            'TV cabinet': 14,
            'Wardrobe': 15,
            'cabinet': 16,
            'gas stove': 17,
            'sink': 18,
            'refrigerator': 19,
            'airconditioner': 20,
            'bath': 21,
            'bath tub': 22,
            'washing machine': 23,  # TODO
            'squat toilet': 24,
            'urinal': 25,
            'toilet': 26,
            'stairs': 27,

            'elevator': 28,
            'escalator': 29,

            'row chairs': 30,
            'parking spot': 31,
            'wall': 32,
            'curtain wall': 33,
            'railing': 34,

        }
        self.anno_list_all_reverse = {v: k for k, v in self.anno_list_all.items()}

        # foreground(countable) classes
        self.anno_list_noBG = {
            'single door': 0,
            'double door': 1,
            'sliding door': 2,
            'folding door': 3,
            'revolving door': 4,
            'rolling door': 5,

            'window': 6,
            'bay window': 7,
            'blind window': 8,
            'opening symbol': 9,

            'sofa': 10,
            'bed': 11,
            'chair': 12,
            'table': 13,
            'TV cabinet': 14,
            'Wardrobe': 15,
            'cabinet': 16,
            'gas stove': 17,
            'sink': 18,
            'refrigerator': 19,
            'airconditioner': 20,
            'bath': 21,
            'bath tub': 22,
            'washing machine': 23,  # TODO
            'squat toilet': 24,
            'urinal': 25,
            'toilet': 26,
            'stairs': 27,

            'elevator': 28,
            'escalator': 29,
        }
        self.anno_list_noBG_reverse = {v: k for k, v in self.anno_list_noBG.items()}

        # windows and door classes
        self.anno_list_door_wind = {
            'single door': 0,
            'double door': 1,
            'sliding door': 2,

            'window': 3,
            'bay window': 4,
            'blind window': 5,
            'opening symbol': 6,

        }
        self.anno_list_door_wind_reverse = {v: k for k, v in self.anno_list_door_wind.items()}


