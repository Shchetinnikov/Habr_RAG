import json

path = "./config.json"


with open(path, 'r') as file:
    config = json.load(file)


if __name__ == "__main__":
    print(config)