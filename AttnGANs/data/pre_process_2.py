import pickle
from sklearn.model_selection import train_test_split

with open("./filenames.txt", "r") as f:
    filenames = [line.rstrip() for line in f.readlines()]

test = ['001.Black_footed_Albatross/Black_Footed_Albatross_0077_796114',
        '002.Laysan_Albatross/Laysan_Albatross_0076_671',
        '003.Sooty_Albatross/Sooty_Albatross_0045_1162',
        '004.Groove_billed_Ani/Groove_Billed_Ani_0101_1700',
        '005.Crested_Auklet/Crested_Auklet_0061_794904',
        '006.Least_Auklet/Least_Auklet_0032_795068',
        '007.Parakeet_Auklet/Parakeet_Auklet_0025_795975',
        '008.Rhinoceros_Auklet/Rhinoceros_Auklet_0012_2161',
        '009.Brewer_Blackbird/Brewer_Blackbird_0035_2611',
        '010.Red_winged_Blackbird/Red_Winged_Blackbird_0088_4007']

others = filenames
for i in range(len(main_test)):
    others.remove(main_test[i])

train, val = train_test_split(others, test_size=0.3)

with open("./train/filenames.pickle","wb") as f:
    pickle.dump(train, f)

with open("./val/filenames.pickle","wb") as f:
    pickle.dump(val, f)

with open("./test/filenames.pickle","wb") as f:
    pickle.dump(test, f)
