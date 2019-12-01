import re

regex = re.compile("a_.*_2")
if regex.match("a_1_2"):
    print("yes")