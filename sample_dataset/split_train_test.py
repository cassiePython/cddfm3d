import os
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split traning and testing dataset")
    parser.add_argument("--interval", type=int, default=3800, help="# of the traning images")
    args = parser.parse_args()

    imgs = os.listdir("Images")
    random.shuffle(imgs)

    interval = args.interval
    train_list = imgs[:3800]
    test_list = imgs[3800:]

    train_list.sort()
    test_list.sort()

    with open("train_list.txt",'w') as fw:
        for path in train_list:
            line = path + "\n"
            fw.write(line)

    with open("test_list.txt",'w') as fw:
        for path in test_list:
            line = path + "\n"
            fw.write(line)
    print ("Finish spliting the traning and testing dataset!")
        
