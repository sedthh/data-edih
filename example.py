from edih import EDIH_GPT

if __name__ == "__main__":
    example = EDIH_GPT("Ma {DATE::%Y.%m.%d %A} van, {DATE::%H:%M:%S} óra.")
    while True:
        print("ASSISTANT:", example.chat(input("USER: ")))
