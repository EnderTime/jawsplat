from model.utils.dataloader import DataLoader


if __name__ == "__main__":
    dl = DataLoader(r"resrc\fox\points\val.json", 3)
    print(dl.img_size)
    print(type(dl.img_size))
    for k in dl:
        print(f"==={k}===")