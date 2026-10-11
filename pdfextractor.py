import json

import pymupdf


def extract_questions(path: str) -> list[str]:
    ## here we want to extract question give the file path
    questions = []
    doc = pymupdf.open(f"{path}")
    text = ""
    for page in doc:
        text += page.get_text()

    i = 1

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
            if text[candidate - 1] == "\n" or candidate == 0:
                nextpos = candidate
                break

        ##print("__" * 70)
        ##print(text[curpos:nextpos])
        questions.append(text[curpos:nextpos])
        curpos = nextpos
        ###print(i, curpos, nextpos)
        i += 1

    mf = text.find("\n30.")

    ###print("__"*70)

    ##print(text[mf:])

    questions.append(text[mf:])

    ### we need to find the text encoded in diagrams as per number only lines
    ###
    filtered_questions = []
    for question in questions:
        filtered = "\n".join(
            line for line in question.splitlines() if not line.strip().isdigit()
        )
        filtered_questions.append(filtered)
    with open(f"{path}.json", "w", encoding="utf-8") as fillip:
        json.dump(filtered_questions, fillip, ensure_ascii=False, indent=2)

    return f"{path}.json"


if __name__ == "__main__":
    result = extract_questions("AMC_2024_Intermediate.pdf")
    print(result)
