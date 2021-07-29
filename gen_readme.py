with open("./docs/README_HEADER.md", "r", encoding="utf-8") as fh:
    long_description_1 = fh.read()

with open("./docs/md/README.md", "r", encoding="utf-8") as fh:
    long_description_2 = fh.read()

long_description = long_description_1 + long_description_2

with open("README.md", "w", encoding="utf-8") as fh:
    fh.write(long_description)
