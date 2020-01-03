boynton_colors = dict(blue=(0,0,255),
    red=(255,0,0),
    green=(0,255,0),
    yellow=(255,255,0),
    magenta=(255,0,255),
    pink=(255,128,128),
    gray=(128,128,128),
    brown=(128,0,0),
    orange=(255,128,0))

kelly_colors = dict(vivid_yellow=(255, 179, 0),
                    strong_purple=(128, 62, 117),
                    vivid_orange=(255, 104, 0),
                    very_light_blue=(166, 189, 215),
                    vivid_red=(193, 0, 32),
                    grayish_yellow=(206, 162, 98),
                    medium_gray=(129, 112, 102),

                    # these aren't good for people with defective color vision:
                    vivid_green=(0, 125, 52),
                    strong_purplish_pink=(246, 118, 142),
                    strong_blue=(0, 83, 138),
                    strong_yellowish_pink=(255, 122, 92),
                    strong_violet=(83, 55, 122),
                    vivid_orange_yellow=(255, 142, 0),
                    strong_purplish_red=(179, 40, 81),
                    vivid_greenish_yellow=(244, 200, 0),
                    strong_reddish_brown=(127, 24, 13),
                    vivid_yellowish_green=(147, 170, 0),
                    deep_yellowish_brown=(89, 51, 21),
                    vivid_reddish_orange=(241, 58, 19),
                    dark_olive_green=(35, 44, 22))

bc = boynton_colors.values()
kc = kelly_colors.values()

from itertools import chain
bc1 = list(chain.from_iterable(boynton_colors.values()))
kc1 = list(chain.from_iterable(kelly_colors.values()))

print(type(bc1))
print(type(kc))
print(bc1)
print(kc)
#high_contrast_colors = boynton_colors.values() + kelly_colors.values()
#high_contrast_colors = boynton_colors.values()
high_contrast_colors = bc1 + kc1
hc_perm = [ 0,  5, 28, 26, 12, 11,  4,  8, 25, 22,  3,  1, 20, 19, 27, 13, 24,
       17, 16, 15,  7, 14, 21, 18, 23,  2, 10,  9,  6]
high_contrast_colors = [high_contrast_colors[i] for i in hc_perm]
print(high_contrast_colors)
