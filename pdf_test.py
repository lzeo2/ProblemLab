import pymupdf
doc = pymupdf.open("AMC_2024_Intermediate.pdf")
text = ""
for page in doc:
    text += page.get_text()


#for i in range 1-30, start by lookking for problem number 1. then afterwards skim throiugh until you get to problem 2. Then, the space in between the start of p1 and p2 is the p1
i = 1

with open(f"sample.txt", "w", encoding="utf-8") as f:
    f.write("")

curpos = text.find("1.")

while i < 30:
    candidates = []

    ## start with problem 1
    nexts = i + 1
    ###print(f"Problem {i} to {nexts}")
    nextpos = text.find(f"{nexts}.", curpos + 1)

    while nextpos != -1:
        candidates.append(nextpos)
        nextpos = text.find(f"{nexts}.", nextpos + 1)

    ## process candidates, checking if at the pos 1 char before and 1 char after is both empty
    for candidate in candidates:
        if text[candidate-1] == "\n" or candidate == 0:
            nextpos = candidate
            break


    print("__"*70)
    print(text[curpos:nextpos])
    with open(f"sample.txt", "a", encoding="utf-8") as f:
        f.write(text[curpos:nextpos])
    curpos = nextpos
    ###print(i, curpos, nextpos)
    i += 1


mf = text.find("\n30.")

###print("__"*70)

##print(text[mf:])
with open(f"sample.txt", "a", encoding="utf-8") as f:
    f.write(text[mf:])


### we need to find the text encoded in diagrams as per number only lines
###
with open("sample.txt", "r", encoding="utf-8") as file:
    with open("nonumbersinthis.txt", "a", encoding="utf-8") as filed:
        for line in file:
            if not line.strip().isdigit():
                filed.write(line)

print()


)
