if __name__ == "__main__":

    for i in range(10000, 10010, 1):
        with open(f"params_{i}.txt", "w") as f:
            f.write(str(i))
    