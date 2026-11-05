import random

# (a) Initialize the list of unique names.
nameLst = [
    "Kevin", "Ada", "Jenny", "Ade", "Betty", "Sam",
    "Kenny", "Beta", "Berry", "Terry", "Nathan", "Jess"
]

# (b) Generate one random age for each name.
ageLst = [random.randint(18, 21) for name in nameLst]

# (c) Combine the names and ages into tuples.
infoLst = list(zip(nameLst, ageLst))
print(infoLst)

# (d) Sort in ascending alphabetical order by name.
nameSortedLst = sorted(infoLst, key=lambda item: item[0])
print(nameSortedLst)

# (e) Sort by descending age, then ascending name.
ageSortedLst = sorted(infoLst, key=lambda item: (-item[1], item[0]))
print(ageSortedLst)

# Sample test results (random ages may vary):
#
# Test run 1:
# nameLst = ['Kevin', 'Ada', 'Jenny', 'Ade', 'Betty', 'Sam', 'Kenny',
#            'Beta', 'Berry', 'Terry', 'Nathan', 'Jess']
# [('Kevin', 19), ('Ada', 18), ('Jenny', 20), ('Ade', 18), ('Betty', 21), ('Sam', 21), ('Kenny', 21), ('Beta', 21), ('Berry', 19), ('Terry', 18), ('Nathan', 21), ('Jess', 18)]
# [('Ada', 18), ('Ade', 18), ('Berry', 19), ('Beta', 21), ('Betty', 21), ('Jenny', 20), ('Jess', 18), ('Kenny', 21), ('Kevin', 19), ('Nathan', 21), ('Sam', 21), ('Terry', 18)]
# [('Beta', 21), ('Betty', 21), ('Kenny', 21), ('Nathan', 21), ('Sam', 21), ('Jenny', 20), ('Berry', 19), ('Kevin', 19), ('Ada', 18), ('Ade', 18), ('Jess', 18), ('Terry', 18)]
#
# Test run 2:
# nameLst = []
# []
# []
# []
#
# Test run 3:
# nameLst = ['Liam', 'Olivia', 'Noah', 'Emma', 'Ava', 'Elijah', 'Sophia', 'Lucas']
# [('Liam', 19), ('Olivia', 19), ('Noah', 20), ('Emma', 21), ('Ava', 18), ('Elijah', 18), ('Sophia', 21), ('Lucas', 20)]
# [('Ava', 18), ('Elijah', 18), ('Emma', 21), ('Liam', 19), ('Lucas', 20), ('Noah', 20), ('Olivia', 19), ('Sophia', 21)]
# [('Emma', 21), ('Sophia', 21), ('Lucas', 20), ('Noah', 20), ('Liam', 19), ('Olivia', 19), ('Ava', 18), ('Elijah', 18)]
